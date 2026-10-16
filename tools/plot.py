"""Dashboard for one kf_demo run.

Usage:
  ./build/kf_demo [options] > out.csv
  .venv/bin/python tools/plot.py out.csv [--title "q=0.05"] [--save out.png]

Panels:
  trajectory        truth, measurements, estimate, 2-sigma ellipses
  position error    error with the +-2 sigma band the filter claims
  velocity error    same for vx (never measured, only inferred)
  uncertainty       sigma = sqrt(P diagonal): how sure the filter is
  Kalman gain       how much one measurement moves the estimate
  NIS               how surprising each measurement was
Orange background = target turning, gray background = no measurements.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

NIS_95 = 5.99  # chi-square with 2 dof: 95 % of NIS values are below this if R, Q are honest


def load(path_or_file):
    return np.genfromtxt(path_or_file, delimiter=",", names=True)


def summary(d) -> dict:
    err2 = (d["x_est"] - d["x_true"]) ** 2 + (d["y_est"] - d["y_true"]) ** 2
    meas = d["has_z"] > 0
    meas2 = (d["z_x"][meas] - d["x_true"][meas]) ** 2 + (d["z_y"][meas] - d["y_true"][meas]) ** 2
    return {
        "rmse": float(np.sqrt(err2.mean())),
        "rmse_sensor": float(np.sqrt(meas2.mean())) if meas.any() else float("nan"),
        "nis": float(np.nanmean(d["nis"])),
        "nees": float(d["nees"].mean()),
        "inside": float(np.mean(np.abs(d["x_est"] - d["x_true"]) <= 2 * np.sqrt(d["P_xx"]))),
    }


def cov_ellipse(pxx, pyy, pxy, n_sigma=2.0):
    """width, height, angle [deg] of the n-sigma ellipse of [[pxx, pxy], [pxy, pyy]]."""
    vals, vecs = np.linalg.eigh(np.array([[pxx, pxy], [pxy, pyy]]))
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))  # direction of the bigger eigenvalue
    width, height = 2 * n_sigma * np.sqrt(np.maximum(vals[::-1], 0))
    return width, height, angle


def spans(t, mask, min_len=1):
    """[(t_start, t_end)] for each run of True in mask that is at least min_len samples long."""
    out, start, n = [], None, 0
    for ti, m in zip(t, mask):
        if m:
            start = ti if start is None else start
            n += 1
        elif start is not None:
            if n >= min_len:
                out.append((start, ti))
            start, n = None, 0
    if start is not None and n >= min_len:
        out.append((start, t[-1]))
    return out


def shade_events(ax, d, legend=False):
    for i, (a, b) in enumerate(spans(d["t"], d["turning"] > 0)):
        ax.axvspan(a, b, color="tab:orange", alpha=0.18, lw=0, label="turn" if legend and i == 0 else None)
    for i, (a, b) in enumerate(spans(d["t"], d["has_z"] == 0, min_len=3)):
        ax.axvspan(a, b, color="gray", alpha=0.25, lw=0, label="no sensor" if legend and i == 0 else None)


def error_band(ax, d, axis, legend=False):
    """estimate - truth for one state variable, with the filter's own +-2 sigma claim."""
    est, true, var = {"x": ("x_est", "x_true", "P_xx"), "vx": ("vx_est", "vx_true", "P_vxvx")}[axis]
    err = d[est] - d[true]
    sigma2 = 2 * np.sqrt(d[var])
    shade_events(ax, d, legend)
    ax.fill_between(d["t"], -sigma2, sigma2, color="tab:blue", alpha=0.2, label="filter's ±2σ" if legend else None)
    ax.plot(d["t"], err, color="tab:red", lw=1, label="real error" if legend else None)
    ax.axhline(0, color="k", lw=0.5)
    # zoom past the start-up transient (huge initial P), otherwise it squashes everything else
    late = d["t"] > 0.15 * d["t"][-1]
    top = 1.3 * max(np.max(np.abs(err[late])), np.max(sigma2[late]))
    ax.set_ylim(-top, top)
    return err, sigma2


def dashboard(d, title=""):
    s = summary(d)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # ---- trajectory
    ax = axes[0, 0]
    ax.plot(d["z_x"], d["z_y"], ".", color="gray", ms=3, alpha=0.5, label="measurements")
    ax.plot(d["x_true"], d["y_true"], "k-", lw=1.5, label="truth")
    ax.plot(d["x_est"], d["y_est"], "-", color="tab:blue", lw=1.5, label="estimate")
    step = max(1, int(round(1.0 / (d["t"][1] - d["t"][0]))))  # one ellipse per second
    for i in range(0, len(d), step):
        w, h, a = cov_ellipse(d["P_xx"][i], d["P_yy"][i], d["P_xy"][i])
        ax.add_patch(Ellipse((d["x_est"][i], d["y_est"][i]), w, h, angle=a, fill=False, color="tab:blue", lw=0.8))
    ax.set_title("trajectory (ellipses: 2σ, one per second)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)

    # ---- errors
    ax = axes[0, 1]
    err, sigma2 = error_band(ax, d, "x", legend=True)
    ax.set_title(f"x position error: {np.mean(np.abs(err) <= sigma2):.0%} inside ±2σ (honest: ~95%)")
    ax.set_ylabel("m")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 2]
    err, sigma2 = error_band(ax, d, "vx")
    ax.set_title(f"vx velocity error (never measured): {np.mean(np.abs(err) <= sigma2):.0%} inside ±2σ")
    ax.text(0.01, 0.02, "start-up transient clipped", transform=ax.transAxes, fontsize=7, color="gray")
    ax.set_ylabel("m/s")

    # ---- uncertainty
    ax = axes[1, 0]
    shade_events(ax, d)
    ax.semilogy(d["t"], np.sqrt(d["P_xx"]), label="σ_x  [m]")
    ax.semilogy(d["t"], np.sqrt(d["P_vxvx"]), label="σ_vx [m/s]")
    ax.set_title("filter's uncertainty  sqrt(P diagonal)")
    ax.legend(fontsize=8)

    # ---- gain
    ax = axes[1, 1]
    shade_events(ax, d)
    meas = d["has_z"] > 0
    ax.semilogy(d["t"], np.where(meas, d["K_pos"], np.nan), label="K_pos: x moves per 1 m of surprise")
    ax.semilogy(d["t"], np.where(meas, d["K_vel"], np.nan), label="K_vel: vx moves per 1 m of surprise")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_title("Kalman gain (K_pos: 1 = copy sensor, 0 = ignore it)")
    ax.legend(fontsize=8)

    # ---- NIS
    ax = axes[1, 2]
    shade_events(ax, d)
    ax.plot(d["t"], d["nis"], ".", ms=3, color="tab:purple")
    ax.axhline(2, color="k", lw=1, label="expected mean 2")
    ax.axhline(NIS_95, color="k", lw=1, ls="--", label="95% line 5.99")
    above = np.nanmean(d["nis"] > NIS_95)
    ax.set_yscale("log")
    ax.set_title(f"NIS (surprise): mean {s['nis']:.2f}, {above:.0%} above line (honest: ~5%)")
    ax.legend(fontsize=8)

    for ax in axes.flat[1:]:
        ax.set_xlabel("t [s]")

    fig.suptitle(
        f"{title}   |   position RootMSqErr: filter {s['rmse']:.2f} m vs raw sensor {s['rmse_sensor']:.2f} m"
        f"   |   mean NIS {s['nis']:.2f} (≈2)   mean NEES {s['nees']:.2f} (≈4)",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default="out.csv")
    p.add_argument("--title", default="")
    p.add_argument("--save", help="write PNG instead of opening a window")
    a = p.parse_args()

    fig = dashboard(load(a.csv), a.title or a.csv)
    if a.save:
        fig.savefig(a.save, dpi=110)
    else:
        plt.show()


if __name__ == "__main__":
    main()
