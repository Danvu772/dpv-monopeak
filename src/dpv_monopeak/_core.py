from __future__ import annotations

from collections.abc import Callable

import numpy as np

LeftChopFn = Callable[[np.ndarray, int], int]


def _find_left_chop(i: np.ndarray, peak_idx: int) -> int:
    # Quarter window = i[0 : peak_idx // 2]. The max of this window is the
    # shoulder of the rising baseline hump that precedes the Faradaic peak.
    quarter_end = peak_idx // 2
    if quarter_end == 0:
        return 0
    return int(np.argmax(i[:quarter_end]))


def compute_dpv(
    v: np.ndarray,
    i: np.ndarray,
    left_chop_fn: LeftChopFn = _find_left_chop,
) -> dict | None:
    """
    Compute all DPV features and intermediates for a single scan.

    Assumes exactly one dominant Faradaic peak. The left baseline cutoff
    is determined by *left_chop_fn* (default: ``_find_left_chop``).

    Parameters
    ----------
    v : np.ndarray
        Voltage array, shape (N,), in volts.
    i : np.ndarray
        Current array, shape (N,), same units throughout (e.g. µA).
    left_chop_fn : callable(i, peak_idx) -> int, optional
        Returns the index below which the left-anchor search is suppressed.
        Signature: ``(i: np.ndarray, peak_idx: int) -> int``.

    Returns
    -------
    dict or None
        On success, a dict with keys:

        Features
        ~~~~~~~~
        X, Height, Area, Width, YOffset, MaxSlope, MinSlope, SumSlope

        Intermediates (for external plotting)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        peak_idx, left_bound, right_bound, baseline, i_corrected,
        half_max, left_hw_idx, right_hw_idx, di, v_mid,
        max_slope_idx, min_slope_idx

        Returns ``None`` when no valid single peak is found.
    """
    if len(v) != len(i):
        return None
    if len(v) < 5:
        return None
    if np.any(np.isnan(v)) or np.any(np.isnan(i)):
        return None
    if np.ptp(i) == 0:
        return None
    if np.any(np.diff(v) == 0):
        return None

    peak_idx     = int(np.argmax(i))
    peak_voltage = v[peak_idx]

    left_chop = left_chop_fn(i, peak_idx)

    if peak_idx >= len(v) - 3 or peak_idx <= left_chop:
        return None

    left_bound  = left_chop + int(np.argmin(i[left_chop:peak_idx]))
    right_bound = (peak_idx + 1) + int(np.argmin(i[peak_idx + 1:]))

    if left_bound >= peak_idx:
        return None
    if right_bound <= peak_idx:
        return None
    if left_bound >= right_bound:
        return None

    left_anchor  = (v[left_bound],  i[left_bound])
    right_anchor = (v[right_bound], i[right_bound])

    if right_anchor[0] == left_anchor[0]:
        return None
    bl_slope    = (right_anchor[1] - left_anchor[1]) / (right_anchor[0] - left_anchor[0])
    baseline    = left_anchor[1] + bl_slope * (v - left_anchor[0])
    y_offset    = baseline[peak_idx]
    i_corrected = i - baseline
    peak_height = i_corrected[peak_idx]
    if peak_height <= 0:
        return None
    peak_area   = np.trapezoid(
        i_corrected[left_bound:right_bound + 1],
        v[left_bound:right_bound + 1],
    )
    if peak_area <= 0:
        return None

    half_max     = peak_height / 2
    left_half    = np.where(i_corrected[:peak_idx] <= half_max)[0]
    right_half   = np.where(i_corrected[peak_idx:] <= half_max)[0]
    left_hw_idx  = int(left_half[-1])            if len(left_half)  > 0 else left_bound
    right_hw_idx = int(right_half[0]) + peak_idx if len(right_half) > 0 else right_bound
    if len(left_half) == 0 or len(right_half) == 0:
        peak_width = np.nan
    else:
        peak_width = v[right_hw_idx] - v[left_hw_idx]
        if peak_width <= 0:
            return None

    di            = np.diff(i_corrected) / np.diff(v)
    v_mid         = (v[:-1] + v[1:]) / 2
    max_slope_idx = left_bound + int(np.argmax(di[left_bound:peak_idx]))
    min_slope_idx = peak_idx   + int(np.argmin(di[peak_idx:right_bound]))
    max_slope     = float(di[max_slope_idx]) if peak_idx > left_bound  else np.nan
    min_slope     = float(di[min_slope_idx]) if right_bound > peak_idx else np.nan
    sum_slope     = (max_slope + abs(min_slope)
                     if not (np.isnan(max_slope) or np.isnan(min_slope))
                     else np.nan)

    return {
        "X":        peak_voltage,
        "Height":   peak_height,
        "Area":     peak_area,
        "Width":    peak_width,
        "YOffset":  y_offset,
        "MaxSlope": max_slope,
        "MinSlope": min_slope,
        "SumSlope": sum_slope,
        "peak_idx":      peak_idx,
        "left_bound":    left_bound,
        "right_bound":   right_bound,
        "baseline":      baseline,
        "i_corrected":   i_corrected,
        "half_max":      half_max,
        "left_hw_idx":   left_hw_idx,
        "right_hw_idx":  right_hw_idx,
        "di":            di,
        "v_mid":         v_mid,
        "max_slope_idx": max_slope_idx,
        "min_slope_idx": min_slope_idx,
    }


def extract_dpv_features(
    v: np.ndarray,
    i: np.ndarray,
    left_chop_fn: LeftChopFn = _find_left_chop,
) -> dict | None:
    """
    Extract the 8 DPV features for a single scan.

    Parameters
    ----------
    v : np.ndarray
        Voltage array (V).
    i : np.ndarray
        Current array, same length as *v*.
    left_chop_fn : callable(i, peak_idx) -> int, optional
        Left-chop strategy. See ``compute_dpv``.

    Returns
    -------
    dict or None
        Keys: X, Height, Area, Width, YOffset, MaxSlope, MinSlope,
        SumSlope. Returns ``None`` when no valid peak is found.

    See Also
    --------
    compute_dpv : returns features plus plotting intermediates.
    """
    result = compute_dpv(v, i, left_chop_fn=left_chop_fn)
    if result is None:
        return None
    return {k: result[k]
            for k in ("X", "Height", "Area", "Width",
                      "YOffset", "MaxSlope", "MinSlope", "SumSlope")}
