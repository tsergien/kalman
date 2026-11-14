// Constant-velocity Kalman filter demo.
//
// Simulates a target moving in 2D, observes its position with a noisy sensor,
// runs KalmanFilter<4> (state [x, y, vx, vy]) and prints CSV to stdout.
// A short summary (errors, NIS, NEES) goes to stderr.
//
//   ./build/kf_demo [options] > out.csv
//   .venv/bin/python tools/plot.py out.csv
//
// Two kinds of knobs (see --help):
//   world  - what really happens  (q-true, sigma-z, turn, gap)
//   filter - what the filter BELIEVES happens (q, r, p0)
// Intuition comes from making them disagree.

#include "kf/kalman.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <string_view>

using kf::KalmanFilter;
using kf::Matrix;

namespace {

struct Config
{
    // world (truth)
    double dt = 0.1;          // time step [s]
    double duration = 60.0;   // [s]
    double q_true = 0.05;     // variance of the target's random acceleration [(m/s^2)^2]
    double sigma_z = 1.0;     // real sensor noise std [m]
    double turn_deg = 90.0;   // each of the two turns (see main), 0 = no turns
    double gap_start = 18.0;  // sensor outage [gap_start, gap_end) s (equal values = no outage)
    double gap_end = 21.0;
    int drop_every = 0;       // also drop every k-th measurement (0 = never)
    unsigned seed = 1;

    // filter (beliefs)
    double q = -1.0;          // accel variance used in Q   (< 0: same as q-true)
    double r = -1.0;          // sensor std used in R       (< 0: same as sigma-z)
    double p0 = 100.0;        // initial variance of every state variable
    double x0_err = 5.0;      // initial position guess is off by this much [m]
};

void print_help()
{
    std::cerr <<
        "usage: kf_demo [options] > out.csv\n"
        "\n"
        "world (truth):\n"
        "  --q-true V     random acceleration variance of the target   (default 0.05)\n"
        "  --sigma-z V    real sensor noise std [m]                    (default 1.0)\n"
        "  --turn DEG     two turns: left t=10..12, right t=27..31, 0 = none (default 90)\n"
        "  --gap A B      sensor outage from A to B seconds            (default 18 21)\n"
        "  --drop K       drop every K-th measurement, 0 = none        (default 0)\n"
        "  --dt V         time step [s]                                (default 0.1)\n"
        "  --duration V   simulated time [s]                           (default 60)\n"
        "  --seed N       random seed                                  (default 1)\n"
        "\n"
        "filter (beliefs):\n"
        "  --q V          process noise in Q   (default: = q-true)  big -> follows measurements\n"
        "  --r V          sensor std in R      (default: = sigma-z) big -> follows prediction\n"
        "  --p0 V         initial variance     (default 100)        small -> trusts bad x0 too long\n"
        "  --x0-err V     initial position error [m]               (default 5)\n";
}

[[noreturn]] void fail(std::string_view msg)
{
    std::cerr << "kf_demo: " << msg << "  (see --help)\n";
    std::exit(2);
}

Config parse(int argc, char** argv)
{
    Config c;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view key = argv[i];
        if (key == "--help" || key == "-h")
        {
            print_help();
            std::exit(0);
        }

        // every option takes one number, --gap takes two
        auto next = [&]() -> double
        {
            if (i + 1 >= argc) fail(std::string("missing value for ") + std::string(key));
            return std::stod(argv[++i]);
        };

        if      (key == "--q-true")   c.q_true = next();
        else if (key == "--sigma-z")  c.sigma_z = next();
        else if (key == "--turn")     c.turn_deg = next();
        else if (key == "--gap")      { c.gap_start = next(); c.gap_end = next(); }
        else if (key == "--drop")     c.drop_every = static_cast<int>(next());
        else if (key == "--dt")       c.dt = next();
        else if (key == "--duration") c.duration = next();
        else if (key == "--seed")     c.seed = static_cast<unsigned>(next());
        else if (key == "--q")        c.q = next();
        else if (key == "--r")        c.r = next();
        else if (key == "--p0")       c.p0 = next();
        else if (key == "--x0-err")   c.x0_err = next();
        else fail(std::string("unknown option ") + std::string(key));
    }
    if (c.q < 0) c.q = c.q_true;
    if (c.r < 0) c.r = c.sigma_z;
    return c;
}

// discrete white-noise acceleration: acceleration a ~ N(0, q) is constant during one step,
// position moves a*dt^2/2, velocity a*dt  =>  per axis Q = q * [dt^4/4  dt^3/2; dt^3/2  dt^2]
Matrix<4, 4> make_Q(double q, double dt)
{
    const double a = dt * dt * dt * dt / 4;
    const double b = dt * dt * dt / 2;
    const double c = dt * dt;
    return q * Matrix<4, 4>{a, 0, b, 0,
                            0, a, 0, b,
                            b, 0, c, 0,
                            0, b, 0, c};
}

}  // namespace

int main(int argc, char** argv)
{
    const Config c = parse(argc, argv);
    const double dt = c.dt;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // model: what the filter believes
    const Matrix<4, 4> F{1, 0, dt, 0,
                         0, 1, 0, dt,
                         0, 0, 1, 0,
                         0, 0, 0, 1};
    const Matrix<2, 4> H{1, 0, 0, 0,
                         0, 1, 0, 0};
    const Matrix<4, 4> Q = make_Q(c.q, dt);
    const Matrix<2, 2> R{c.r * c.r, 0,
                         0, c.r * c.r};

    std::mt19937 rng(c.seed);
    std::normal_distribution<double> std_normal(0.0, 1.0);

    // state layout everywhere (F, H, Q, P): [x, y, vx, vy]
    // truth starts at the origin, vx = 2 m/s (right), vy = 1 m/s (up). filter starts at a wrong position, zero velocity.
    Matrix<4, 1> truth{0, 0, 2.0, 1.0}; // x, y, vx, vy
    KalmanFilter<4> kf(Matrix<4, 1>{c.x0_err, -c.x0_err, 0, 0}, c.p0 * Matrix<4, 4>::identity());

    // two turns by turn_deg: a sharp left (2 s), 15 s straight, a slower right (4 s).
    // same angle in double the time = half the turn rate = half the unexpected acceleration.
    struct Turn
    {
        double start;     // [s]
        double duration;  // [s]
        double direction; // +1 left, -1 right
    };
    const Turn turns[] = {{10.0, 2.0, +1.0}, {27.0, 4.0, -1.0}};
    const double sigma_a_true = std::sqrt(c.q_true);

    // summary statistics
    double sum_err2 = 0, sum_meas_err2 = 0, sum_nis = 0, sum_nees = 0;
    int n_meas = 0, n_inside = 0;
    const int steps = static_cast<int>(std::lround(c.duration / dt));

    std::cout << std::setprecision(8);
    std::cout << "t,x_true,y_true,vx_true,vy_true,turning,has_z,z_x,z_y,"
                 "x_est,y_est,vx_est,vy_est,P_xx,P_yy,P_xy,P_vxvx,P_vyvy,K_pos,K_vel,nis,nees\n";

    for (int k = 1; k <= steps; ++k)
    {
        const double t = k * dt;

        // ---- world: random acceleration + optional turn (acceleration perpendicular to velocity)
        double ax = sigma_a_true * std_normal(rng);
        double ay = sigma_a_true * std_normal(rng);
        bool turning = false;
        for (const Turn& turn : turns)
        {
            if (c.turn_deg == 0.0 || t <= turn.start || t > turn.start + turn.duration) continue;
            const double rate = turn.direction * c.turn_deg * std::numbers::pi / 180.0 / turn.duration; // [rad/s]
            ax += -rate * truth(3, 0);
            ay +=  rate * truth(2, 0);
            turning = true;
        }
        truth(0, 0) += truth(2, 0) * dt + ax * dt * dt / 2;
        truth(1, 0) += truth(3, 0) * dt + ay * dt * dt / 2;
        truth(2, 0) += ax * dt;
        truth(3, 0) += ay * dt;

        const bool in_gap = t >= c.gap_start && t < c.gap_end;
        const bool dropped = c.drop_every > 0 && k % c.drop_every == 0;
        const bool has_z = !in_gap && !dropped;
        const Matrix<2, 1> z = H * truth + c.sigma_z * Matrix<2, 1>{std_normal(rng), std_normal(rng)};

        // ---- filter
        kf.predict(F, Q);

        double K_pos = 0, K_vel = 0, nis = nan;
        if (has_z)
        {
            // diagnostics from the prior (the filter computes the same things inside update)
            const auto x_prior = kf.state();
            const auto P_prior = kf.covariance();
            const auto S = H * P_prior * H.transpose() + R;
            const auto y = z - H * x_prior;
            const auto K = S.solve(H * P_prior).transpose();

            nis = (y.transpose() * S.solve(y))(0, 0); // y^T S^-1 y: how surprising was z?
            K_pos = K(0, 0); // how much x moves per 1 m of x-innovation
            K_vel = K(2, 0); // how much vx moves per 1 m of x-innovation

            kf.update(z, H, R);
        }

        const auto x = kf.state();
        const auto P = kf.covariance();
        const auto e = truth - x;
        const double nees = (e.transpose() * P.solve(e))(0, 0); // e^T P^-1 e: is P honest?

        // ---- statistics
        sum_err2 += e(0, 0) * e(0, 0) + e(1, 0) * e(1, 0);
        sum_nees += nees;
        if (std::abs(e(0, 0)) <= 2 * std::sqrt(P(0, 0))) ++n_inside;
        if (has_z)
        {
            const double mx = z(0, 0) - truth(0, 0), my = z(1, 0) - truth(1, 0);
            sum_meas_err2 += mx * mx + my * my;
            sum_nis += nis;
            ++n_meas;
        }

        // ---- csv
        std::cout << t << ',' << truth(0, 0) << ',' << truth(1, 0) << ',' << truth(2, 0) << ',' << truth(3, 0) << ','
                  << (turning ? 1 : 0) << ',' << (has_z ? 1 : 0) << ',' << (has_z ? z(0, 0) : nan) << ',' << (has_z ? z(1, 0) : nan) << ','
                  << x(0, 0) << ',' << x(1, 0) << ',' << x(2, 0) << ',' << x(3, 0) << ','
                  << P(0, 0) << ',' << P(1, 1) << ',' << P(0, 1) << ',' << P(2, 2) << ',' << P(3, 3) << ','
                  << K_pos << ',' << K_vel << ',' << nis << ',' << nees << '\n';
    }

    std::cerr << std::fixed << std::setprecision(3)
              << "position RootMeanSquareError  filter: " << std::sqrt(sum_err2 / steps)
              << " m   raw sensor: " << std::sqrt(sum_meas_err2 / std::max(n_meas, 1)) << " m\n"
              << "x error inside 2 sigma: " << 100.0 * n_inside / steps << " %   (honest P: ~95 %)\n"
              << "mean Normalized Innovation Squared:  " << sum_nis / std::max(n_meas, 1) << "   (honest: ~2, measurement size)\n"
              << "mean Normalized Estimation Error Squared: " << sum_nees / steps << "   (honest: ~4, state size)\n";
}
