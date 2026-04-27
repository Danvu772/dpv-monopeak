"""
Synthetic-peak tests for dpv_monopeak.

Ground truth is analytically known for a Gaussian peak on a flat baseline:
  - X      = mu (peak voltage)
  - Height = amplitude A
  - Width  = FWHM = 2 * sqrt(2 * ln 2) * sigma
  - Area   ≈ A * sigma * sqrt(2 * pi)   (over the full support)

All tests use a Gaussian centred at v=0.0 on a zero baseline unless
noted otherwise. With a flat zero baseline the _find_left_chop heuristic
returns 0 (the quarter window is essentially zero everywhere), so the
left anchor lands at the array edge — a benign no-op.
"""

import numpy as np
import pytest
from dpv_monopeak import compute_dpv, extract_dpv_features

_FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))  # ≈ 2.3548


def _gaussian_scan(
    n: int = 101,
    mu: float = 0.0,
    sigma: float = 0.05,
    amp: float = 100.0,
    v_lo: float = -0.5,
    v_hi: float = 0.5,
    baseline_slope: float = 0.0,
    baseline_intercept: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    v = np.linspace(v_lo, v_hi, n)
    i = amp * np.exp(-0.5 * ((v - mu) / sigma) ** 2)
    i += baseline_slope * v + baseline_intercept
    return v, i


# ── basic feature accuracy ────────────────────────────────────────────────────

def test_returns_dict_on_valid_scan():
    v, i = _gaussian_scan()
    assert compute_dpv(v, i) is not None


def test_peak_voltage_within_one_step():
    v, i = _gaussian_scan(mu=0.0)
    dv = v[1] - v[0]
    result = compute_dpv(v, i)
    assert abs(result["X"] - 0.0) <= dv


def test_peak_height_accuracy():
    v, i = _gaussian_scan(amp=100.0)
    result = compute_dpv(v, i)
    assert abs(result["Height"] - 100.0) / 100.0 < 0.01  # <1 %


def test_peak_width_near_fwhm():
    sigma = 0.05
    analytic_fwhm = _FWHM_FACTOR * sigma  # ≈ 0.1177 V
    v, i = _gaussian_scan(sigma=sigma)
    result = compute_dpv(v, i)
    # Discrete sampling adds at most one voltage step of error on each side.
    dv = v[1] - v[0]
    assert abs(result["Width"] - analytic_fwhm) < 2 * dv


def test_peak_area_near_analytic():
    sigma, amp = 0.05, 100.0
    analytic_area = amp * sigma * np.sqrt(2 * np.pi)  # ≈ 12.53 µA·V
    v, i = _gaussian_scan(sigma=sigma, amp=amp)
    result = compute_dpv(v, i)
    # Trapezoidal rule + truncated tails → allow 5 %
    assert abs(result["Area"] - analytic_area) / analytic_area < 0.05


# ── extract_dpv_features ─────────────────────────────────────────────────────

def test_extract_returns_exactly_8_keys():
    v, i = _gaussian_scan()
    feats = extract_dpv_features(v, i)
    assert feats is not None
    assert set(feats.keys()) == {
        "X", "Height", "Area", "Width",
        "YOffset", "MaxSlope", "MinSlope", "SumSlope",
    }


def test_extract_matches_compute_features():
    v, i = _gaussian_scan()
    full   = compute_dpv(v, i)
    feats  = extract_dpv_features(v, i)
    for k in feats:
        assert feats[k] == full[k]


# ── compute_dpv intermediates ─────────────────────────────────────────────────

def test_compute_contains_intermediates():
    required = {
        "baseline", "i_corrected", "peak_idx",
        "left_bound", "right_bound", "di", "v_mid",
    }
    v, i = _gaussian_scan()
    result = compute_dpv(v, i)
    assert required.issubset(result.keys())


def test_i_corrected_peak_equals_height():
    v, i = _gaussian_scan()
    r = compute_dpv(v, i)
    assert abs(r["i_corrected"][r["peak_idx"]] - r["Height"]) < 1e-10


# ── slope signs ──────────────────────────────────────────────────────────────

def test_max_slope_positive():
    v, i = _gaussian_scan()
    assert compute_dpv(v, i)["MaxSlope"] > 0


def test_min_slope_negative():
    v, i = _gaussian_scan()
    assert compute_dpv(v, i)["MinSlope"] < 0


def test_sum_slope_positive():
    v, i = _gaussian_scan()
    result = compute_dpv(v, i)
    assert result["SumSlope"] == pytest.approx(
        result["MaxSlope"] + abs(result["MinSlope"])
    )


# ── robustness ───────────────────────────────────────────────────────────────

def test_returns_none_when_peak_at_edge():
    v = np.linspace(0, 1, 50)
    i = np.zeros(50)
    i[-1] = 10.0  # peak at the very last sample
    assert compute_dpv(v, i) is None


def test_off_centre_peak():
    """Peak at v=0.2, not centred — should still extract correctly."""
    v, i = _gaussian_scan(mu=0.2, v_lo=-0.1, v_hi=0.5)
    result = compute_dpv(v, i)
    assert result is not None
    dv = v[1] - v[0]
    assert abs(result["X"] - 0.2) <= dv


def test_linear_baseline_subtracted():
    """A tilted baseline should not inflate YOffset unreasonably."""
    v, i = _gaussian_scan(baseline_slope=50.0, baseline_intercept=5.0)
    result = compute_dpv(v, i)
    assert result is not None
    # After subtraction the corrected height should still be near 100
    assert abs(result["Height"] - 100.0) / 100.0 < 0.05


# ── _find_left_chop ──────────────────────────────────────────────────────────

def test_find_left_chop_left_of_peak():
    from dpv_monopeak._core import _find_left_chop
    v, i = _gaussian_scan()
    peak_idx = int(np.argmax(i))
    chop = _find_left_chop(i, peak_idx)
    assert chop < peak_idx


# ── left_chop_fn injection ───────────────────────────────────────────────────

def test_custom_left_chop_fn_is_called():
    """A lambda returning 0 is invoked and still yields a valid result."""
    v, i = _gaussian_scan()
    calls = []

    def recording_chop(arr, p):
        calls.append(p)
        return 0

    result = compute_dpv(v, i, left_chop_fn=recording_chop)
    assert result is not None
    assert len(calls) == 1
    assert calls[0] == int(np.argmax(i))
