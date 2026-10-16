"""Run kf_demo several times, changing ONE knob, and show the runs side by side.

Usage:
  .venv/bin/python tools/compare.py PARAM V1 V2 ... [kf_demo options] [--animate FILE|live] [--save PNG]

  PARAM is any kf_demo option without the dashes: q, r, p0, q-true, sigma-z, turn, drop, ...
  Every other option is passed to kf_demo unchanged, so all runs share the same world.

Examples:
  .venv/bin/python tools/compare.py q 0.0005 0.05 5                    # trust physics vs trust sensor
  .venv/bin/python tools/compare.py r 0.2 1 5 --turn 0                 # filter's opinion of the sensor
  .venv/bin/python tools/compare.py p0 0.01 100 --turn 0 --x0-err 10   # wrong start, over-confident
  .venv/bin/python tools/compare.py q 0.0005 5 --animate live          # animation in a window
  .venv/bin/python tools/compare.py q 0.0005 5 --animate kf.mp4        # save video (ffmpeg) or .gif

Same seed everywhere: the target and the measurements are IDENTICAL in every column,
only the filter's belief changes.
"""

import argparse
import io
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot import NIS_95, cov_ellipse, error_band, load, shade_events, summary  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COLORS = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def run(exe, param, value, extra):
    cmd = [str(exe), *extra, f"--{param}", str(value)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed:\n{res.stderr}")
    return load(io.StringIO(res.stdout))


# ---- static comparison --------------------------------------------------------------

def compare_figure(runs, labels, header):
    n = len(runs)
    fig, axes = plt.subplots(4, n, figsize=(5.2 * n, 13), squeeze=False, sharey="row")

    tops = []
    for col, (d, label) in enumerate(zip(runs, labels)):
        s = summary(d)
        c = COLORS[col % len(COLORS)]

        ax = axes[0, col]
        ax.plot(d["z_x"], d["z_y"], ".", color="gray", ms=2, alpha=0.4)
        ax.plot(d["x_true"], d["y_true"], "k-", lw=1.2, label="truth")
        ax.plot(d["x_est"], d["y_est"], "-", color=c, lw=1.5, label="estimate")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{label}\nRMSE {s['rmse']:.2f} m (sensor {s['rmse_sensor']:.2f})   "
                     f"NIS {s['nis']:.1f}  NEES {s['nees']:.1f}", fontsize=10)
        if col == 0:
            ax.legend(fontsize=8)

        ax = axes[1, col]
        err, sigma2 = error_band(ax, d, "x", legend=(col == 0))
        late = d["t"] > 0.15 * d["t"][-1]
        tops.append(1.3 * max(np.max(np.abs(err[late])), np.max(sigma2[late])))
        ax.set_title(f"x error, {np.mean(np.abs(err) <= sigma2):.0%} inside ±2σ", fontsize=10)
        if col == 0:
            ax.legend(fontsize=7, loc="upper right")

        ax = axes[2, col]
        shade_events(ax, d)
        meas = d["has_z"] > 0
        ax.semilogy(d["t"], np.where(meas, d["K_pos"], np.nan), color=c)
        ax.set_title(f"K_pos (steady ≈ {np.nanmedian(np.where(meas, d['K_pos'], np.nan)):.3f})", fontsize=10)

        ax = axes[3, col]
        shade_events(ax, d)
        ax.semilogy(d["t"], d["nis"], ".", ms=2, color=c)
        ax.axhline(2, color="k", lw=0.8)
        ax.axhline(NIS_95, color="k", lw=0.8, ls="--")
        ax.set_title(f"NIS, {np.nanmean(d['nis'] > NIS_95):.0%} above 5.99 (honest ~5%)", fontsize=10)
        ax.set_xlabel("t [s]")

    # shared y axis: make the worst column fit
    axes[1, 0].set_ylim(-max(tops), max(tops))

    axes[0, 0].set_ylabel("y [m]")
    axes[1, 0].set_ylabel("x error [m]")
    axes[2, 0].set_ylabel("gain")
    axes[3, 0].set_ylabel("NIS")
    fig.suptitle(header, fontsize=12)
    fig.tight_layout()
    return fig


# ---- animation ------------------------------------------------------------------------

def animate(runs, labels, header, speed):
    n = len(runs)
    d0 = runs[0]
    t = d0["t"]
    dt = t[1] - t[0]
    trail = int(round(4.0 / dt))   # show the last 4 s of history
    half = 6.0                     # camera half-width [m]

    # inches * dpi(100) must be whole, even pixel counts, or ffmpeg receives misaligned frames
    fig = plt.figure(figsize=(5.5 * n, 8.0))
    gs = fig.add_gridspec(2, n, height_ratios=[3, 1])
    fig.suptitle(header, fontsize=12)

    artists = []
    for col, (d, label) in enumerate(zip(runs, labels)):
        c = COLORS[col % len(COLORS)]

        ax = fig.add_subplot(gs[0, col])
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11)
        a = {"ax": ax, "d": d}
        a["truth"], = ax.plot([], [], "k-", lw=1.5, label="truth")
        a["meas"], = ax.plot([], [], "o", color="gray", ms=4, alpha=0.6, label="measurements")
        a["last_z"], = ax.plot([], [], "o", color="gold", mec="k", ms=8, label="newest measurement")
        a["est"], = ax.plot([], [], "-", color=c, lw=2, label="estimate")
        a["vel"], = ax.plot([], [], "-", color=c, lw=1, alpha=0.7)
        a["now_true"], = ax.plot([], [], "kx", ms=10, mew=2)
        a["now_est"], = ax.plot([], [], "o", color=c, ms=7)
        a["ell"] = Ellipse((0, 0), 0, 0, fill=False, color=c, lw=1.5, label="2σ ellipse (P)")
        ax.add_patch(a["ell"])
        a["text"] = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9, family="monospace",
                            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        a["status"] = ax.text(0.98, 0.98, "", transform=ax.transAxes, va="top", ha="right", fontsize=12,
                              fontweight="bold")
        if col == 0:
            ax.legend(fontsize=7, loc="lower right")

        ex = fig.add_subplot(gs[1, col])
        err, sigma2 = error_band(ex, d, "x")
        ex.set_ylabel("x error [m]")
        ex.set_xlabel("t [s]")
        a["cursor"] = ex.axvline(0, color="k", lw=1)
        artists.append(a)

    def frame(i):
        lo = max(0, i - trail)
        out = []
        for a in artists:
            d = a["d"]
            a["truth"].set_data(d["x_true"][lo:i + 1], d["y_true"][lo:i + 1])
            a["meas"].set_data(d["z_x"][lo:i + 1], d["z_y"][lo:i + 1])
            a["last_z"].set_data([d["z_x"][i]], [d["z_y"][i]])
            a["est"].set_data(d["x_est"][lo:i + 1], d["y_est"][lo:i + 1])
            # estimated velocity, drawn as "where I think I'll be in 1 s"
            a["vel"].set_data([d["x_est"][i], d["x_est"][i] + d["vx_est"][i]],
                              [d["y_est"][i], d["y_est"][i] + d["vy_est"][i]])
            a["now_true"].set_data([d["x_true"][i]], [d["y_true"][i]])
            a["now_est"].set_data([d["x_est"][i]], [d["y_est"][i]])

            w, h, ang = cov_ellipse(d["P_xx"][i], d["P_yy"][i], d["P_xy"][i])
            a["ell"].set_center((d["x_est"][i], d["y_est"][i]))
            a["ell"].set_width(w)
            a["ell"].set_height(h)
            a["ell"].set_angle(ang)

            # camera: centered between truth and estimate, zooms out when they drift apart
            cx = (d["x_true"][i] + d["x_est"][i]) / 2
            cy = (d["y_true"][i] + d["y_est"][i]) / 2
            dist = np.hypot(d["x_est"][i] - d["x_true"][i], d["y_est"][i] - d["y_true"][i])
            r = max(half, 0.5 * dist + 3)
            a["ax"].set_xlim(cx - r, cx + r)
            a["ax"].set_ylim(cy - r, cy + r)

            err = np.hypot(d["x_est"][i] - d["x_true"][i], d["y_est"][i] - d["y_true"][i])
            k = f"{d['K_pos'][i]:.3f}" if d["has_z"][i] else "  -  "
            a["text"].set_text(f"t      {d['t'][i]:5.1f} s\n"
                               f"K_pos  {k}\n"
                               f"σ_x    {np.sqrt(d['P_xx'][i]):.3f} m\n"
                               f"error  {err:.3f} m")
            if not d["has_z"][i]:
                a["status"].set_text("NO SENSOR")
                a["status"].set_color("dimgray")
            elif d["turning"][i]:
                a["status"].set_text("TURNING")
                a["status"].set_color("darkorange")
            else:
                a["status"].set_text("")
            a["cursor"].set_xdata([d["t"][i]])
            out += [a[k] for k in ("truth", "meas", "last_z", "est", "vel", "now_true", "now_est",
                                   "ell", "text", "status", "cursor")]
        return out

    fig.tight_layout()
    fps = max(1, int(round(speed / dt)))
    anim = FuncAnimation(fig, frame, frames=len(t), interval=1000 / fps, blit=False)
    return fig, anim, fps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("param", help="kf_demo option to vary, without dashes (q, r, p0, q-true, sigma-z, turn, ...)")
    p.add_argument("values", nargs="+", type=float)
    p.add_argument("--exe", default=str(ROOT / "build" / "kf_demo"))
    p.add_argument("--save", help="write the comparison PNG instead of opening a window")
    p.add_argument("--animate", metavar="FILE|live", help="make an animation: .mp4, .gif, or 'live' window")
    p.add_argument("--speed", type=float, default=2.0, help="animation: simulated seconds per real second")
    a, extra = p.parse_known_args()

    runs = [run(a.exe, a.param, v, extra) for v in a.values]
    labels = [f"{a.param} = {v:g}" for v in a.values]
    header = f"varying {a.param}" + (f"   (other options: {' '.join(extra)})" if extra else "")

    if a.animate:
        fig, anim, fps = animate(runs, labels, header, a.speed)
        if a.animate == "live":
            plt.show()
        else:
            writer = PillowWriter(fps=fps) if a.animate.endswith(".gif") else FFMpegWriter(fps=fps, bitrate=2400)
            anim.save(a.animate, writer=writer, dpi=100)
            print(f"wrote {a.animate}")
        return

    fig = compare_figure(runs, labels, header)
    if a.save:
        fig.savefig(a.save, dpi=100)
        print(f"wrote {a.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
