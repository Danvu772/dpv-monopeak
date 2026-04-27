from __future__ import annotations

import numpy as np

from ._core import LeftChopFn, _find_left_chop, compute_dpv

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def plot_dpv(
    v: np.ndarray,
    i: np.ndarray,
    left_chop_fn: LeftChopFn = _find_left_chop,
    ax: tuple | None = None,
) -> tuple:
    """
    Plot DPV features for a single scan.

    Produces a two-panel figure: raw current + baseline (top), baseline-
    corrected current + FWHM annotation (bottom).

    Parameters
    ----------
    v : np.ndarray
        Voltage array (V).
    i : np.ndarray
        Current array, same units throughout (e.g. µA).
    left_chop_fn : callable(i, peak_idx) -> int, optional
        Left-chop strategy passed through to ``compute_dpv``.
    ax : tuple of two Axes, optional
        ``(ax1, ax2)`` to draw into. If *None*, a new figure is created.

    Returns
    -------
    tuple of (Axes, Axes)
        ``(ax1, ax2)`` — top and bottom panels.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If no valid peak is found in the scan.
    """
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )

    r = compute_dpv(v, i, left_chop_fn=left_chop_fn)
    if r is None:
        raise ValueError("No valid peak found in scan.")

    if ax is None:
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    else:
        ax1, ax2 = ax

    dv_tan = 0.05  # tangent line half-width in V

    def _tangent(slope, x0, y0):
        xs = np.array([x0 - dv_tan, x0 + dv_tan])
        return xs, y0 + slope * (xs - x0)

    # ── top panel: raw current, baseline, height, slopes ────────────────────
    ax1.plot(v, i, color="steelblue", linewidth=2, label="DPV current")
    ax1.plot(v, r["baseline"], color="gray", linewidth=1.2, linestyle="--",
             label="baseline")
    ax1.scatter(
        [v[r["left_bound"]], v[r["right_bound"]]],
        [i[r["left_bound"]], i[r["right_bound"]]],
        color="green", zorder=5, label="anchors",
    )
    ax1.axvline(r["X"], color="red", linewidth=0.8, linestyle=":")
    ax1.annotate(
        "",
        xy=(r["X"], i[r["peak_idx"]]),
        xytext=(r["X"], r["YOffset"]),
        arrowprops=dict(arrowstyle="<->", color="red", lw=1.5),
    )
    ax1.text(
        r["X"] + 0.005, r["YOffset"] + r["Height"] / 2,
        f"Height={r['Height']:.1f}", color="red", fontsize=8,
    )
    ax1.scatter(
        [r["X"]], [r["YOffset"]], color="orange", zorder=5,
        label=f"YOffset={r['YOffset']:.1f}\nX={r['X']:.4f} V",
    )

    tx, ty = _tangent(r["MaxSlope"], r["v_mid"][r["max_slope_idx"]], i[r["max_slope_idx"]])
    ax1.plot(tx, ty, color="red", linewidth=1.5,
             label=f"MaxSlope={r['MaxSlope']:.1f}")

    tx, ty = _tangent(r["MinSlope"], r["v_mid"][r["min_slope_idx"]], i[r["min_slope_idx"]])
    ax1.plot(tx, ty, color="purple", linewidth=1.5,
             label=f"MinSlope={r['MinSlope']:.1f}")

    ax1.set_ylabel("Current (µA)")
    ax1.legend(fontsize=7, frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── bottom panel: corrected current, FWHM ───────────────────────────────
    ax2.plot(v, r["i_corrected"], color="steelblue", linewidth=2,
             label="corrected current")
    ax2.axhline(r["half_max"], color="purple", linewidth=0.8, linestyle="--",
                label=f"half max={r['half_max']:.1f}")
    ax2.annotate(
        "",
        xy=(v[r["right_hw_idx"]], r["half_max"]),
        xytext=(v[r["left_hw_idx"]], r["half_max"]),
        arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5),
    )
    ax2.text(
        v[r["left_hw_idx"]], r["half_max"] + r["Height"] * 0.1,
        f"Width={r['Width']:.4f} V", color="purple", fontsize=8,
    )
    ax2.axvline(r["X"], color="red", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Voltage (V)")
    ax2.set_ylabel("Corrected Current (µA)")
    ax2.legend(fontsize=7, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    ax1.figure.tight_layout()
    return ax1, ax2
