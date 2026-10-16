#include "kf/kalman.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

using kf::KalmanFilter;
using kf::Matrix;

namespace {

template <std::size_t N>
void expectSymmetric(const Matrix<N, N>& P, double tol = 1e-12) {
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i + 1; j < N; ++j) EXPECT_NEAR(P(i, j), P(j, i), tol);
}

// Detects whether kf.update(z, H, R) compiles for these argument types.
template <typename KF, typename Z, typename H, typename R>
concept CanUpdate = requires(KF kf, Z z, H h, R r) { kf.update(z, h, r); };

}  // namespace

// ---- compile-time checks ---------------------------------------------------

// M is deduced from the arguments, so one filter accepts sensors of any size...
static_assert(CanUpdate<KalmanFilter<2>, Matrix<1, 1>, Matrix<1, 2>, Matrix<1, 1>>);
static_assert(CanUpdate<KalmanFilter<2>, Matrix<2, 1>, Matrix<2, 2>, Matrix<2, 2>>);
// ...but the shapes must agree with each other and with N.
static_assert(!CanUpdate<KalmanFilter<2>, Matrix<2, 1>, Matrix<3, 2>, Matrix<2, 2>>,
              "z says M=2, H says M=3: must not compile");
static_assert(!CanUpdate<KalmanFilter<2>, Matrix<1, 1>, Matrix<1, 3>, Matrix<1, 1>>,
              "H has 3 columns but N=2: must not compile");

// ---- Step 3: scalar filter -------------------------------------------------

TEST(Kalman1D, PredictAddsProcessNoise) {
    KalmanFilter<1> kf(Matrix<1, 1>{5.0}, Matrix<1, 1>{2.0});
    kf.predict(Matrix<1, 1>{1.0}, Matrix<1, 1>{0.5});
    EXPECT_DOUBLE_EQ(kf.state()(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(kf.covariance()(0, 0), 2.5);
}

TEST(Kalman1D, UpdateWithEqualVariancesAverages) {
    // Prior N(0, 4), measurement 10 with variance 4 -> posterior N(5, 2).
    KalmanFilter<1> kf(Matrix<1, 1>{0.0}, Matrix<1, 1>{4.0});
    kf.update(Matrix<1, 1>{10.0}, Matrix<1, 1>{1.0}, Matrix<1, 1>{4.0});
    EXPECT_NEAR(kf.state()(0, 0), 5.0, 1e-12);
    EXPECT_NEAR(kf.covariance()(0, 0), 2.0, 1e-12);
}

TEST(Kalman1D, PreciseMeasurementDominates) {
    KalmanFilter<1> kf(Matrix<1, 1>{0.0}, Matrix<1, 1>{100.0});
    kf.update(Matrix<1, 1>{3.0}, Matrix<1, 1>{1.0}, Matrix<1, 1>{1e-6});
    EXPECT_NEAR(kf.state()(0, 0), 3.0, 1e-6);
}

TEST(Kalman1D, EstimatesConstantFromNoisyReadings) {
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 1.0);
    const double truth = 7.3;

    KalmanFilter<1> kf(Matrix<1, 1>{0.0}, Matrix<1, 1>{1000.0});
    for (int k = 0; k < 500; ++k) {
        kf.predict(Matrix<1, 1>{1.0}, Matrix<1, 1>{0.0});
        kf.update(Matrix<1, 1>{truth + noise(rng)}, Matrix<1, 1>{1.0}, Matrix<1, 1>{1.0});
    }
    EXPECT_NEAR(kf.state()(0, 0), truth, 0.15);
    // With Q = 0 this is the running mean: variance ~ 1/n.
    EXPECT_NEAR(kf.covariance()(0, 0), 1.0 / 500.0, 1e-4);
}

// ---- Step 4: constant-velocity model ---------------------------------------

TEST(KalmanCV, PredictMovesPositionByVelocity) {
    // state [x, y, vx, vy]
    KalmanFilter<4> kf(Matrix<4, 1>{0, 0, 1, 2}, Matrix<4, 4>::identity());
    const double dt = 0.5;
    Matrix<4, 4> F{1, 0, dt, 0,
                   0, 1, 0, dt,
                   0, 0, 1, 0,
                   0, 0, 0, 1};
    kf.predict(F, Matrix<4, 4>{});
    EXPECT_DOUBLE_EQ(kf.state()(0, 0), 0.5);
    EXPECT_DOUBLE_EQ(kf.state()(1, 0), 1.0);
    // Position uncertainty grows: P_xx = 1 + dt^2, and x/vx become correlated.
    EXPECT_DOUBLE_EQ(kf.covariance()(0, 0), 1.0 + dt * dt);
    EXPECT_DOUBLE_EQ(kf.covariance()(0, 2), dt);
    expectSymmetric(kf.covariance());
}

TEST(KalmanCV, TracksMovingTargetAndStaysConsistent) {
    std::mt19937 rng(7);
    const double sigma_z = 0.5;
    std::normal_distribution<double> meas_noise(0.0, sigma_z);

    const double dt = 0.1;
    Matrix<4, 4> F{1, 0, dt, 0,
                   0, 1, 0, dt,
                   0, 0, 1, 0,
                   0, 0, 0, 1};
    Matrix<2, 4> H{1, 0, 0, 0,
                   0, 1, 0, 0};
    Matrix<2, 2> R{sigma_z * sigma_z, 0, 0, sigma_z * sigma_z};
    Matrix<4, 4> Q = 1e-6 * Matrix<4, 4>::identity();

    Matrix<4, 1> truth{0, 0, 1.0, -0.5};
    KalmanFilter<4> kf(Matrix<4, 1>{}, 100.0 * Matrix<4, 4>::identity());

    for (int k = 0; k < 300; ++k) {
        truth = F * truth;
        kf.predict(F, Q);
        Matrix<2, 1> z = H * truth + Matrix<2, 1>{meas_noise(rng), meas_noise(rng)};
        kf.update(z, H, R);

        expectSymmetric(kf.covariance(), 1e-9);
        for (std::size_t i = 0; i < 4; ++i) EXPECT_GT(kf.covariance()(i, i), 0.0);
    }
    EXPECT_NEAR(kf.state()(0, 0), truth(0, 0), 0.2);
    EXPECT_NEAR(kf.state()(1, 0), truth(1, 0), 0.2);
    EXPECT_NEAR(kf.state()(2, 0), truth(2, 0), 0.1);
    EXPECT_NEAR(kf.state()(3, 0), truth(3, 0), 0.1);
}

// ---- Two sensors with different M on one filter ----------------------------

TEST(KalmanFusion, TwoSensorsWithDifferentSizes) {
    // state [pos, vel], prior N(0, I)
    KalmanFilter<2> kf(Matrix<2, 1>{0, 0}, Matrix<2, 2>::identity());

    // Sensor A (M = 1): position only. z = 2, R = 1.
    //   S = 1 + 1 = 2, K = [0.5; 0]  ->  x = [1, 0], P = diag(0.5, 1)
    kf.update(Matrix<1, 1>{2.0}, Matrix<1, 2>{1, 0}, Matrix<1, 1>{1.0});
    EXPECT_NEAR(kf.state()(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(kf.state()(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(kf.covariance()(0, 0), 0.5, 1e-12);
    EXPECT_NEAR(kf.covariance()(1, 1), 1.0, 1e-12);

    // Sensor B (M = 2): position and velocity. z = [1, 4], R = I.
    //   pos: prior 1 agrees with z = 1 -> stays 1, P = 0.5*1/(0.5+1) = 1/3
    //   vel: K = 1/(1+1) = 0.5 -> 0 + 0.5*4 = 2, P = 1*1/(1+1) = 0.5
    kf.update(Matrix<2, 1>{1.0, 4.0}, Matrix<2, 2>::identity(), Matrix<2, 2>::identity());
    EXPECT_NEAR(kf.state()(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(kf.state()(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(kf.covariance()(0, 0), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(kf.covariance()(1, 1), 0.5, 1e-12);
    EXPECT_NEAR(kf.covariance()(0, 1), 0.0, 1e-12);
    expectSymmetric(kf.covariance());
}
