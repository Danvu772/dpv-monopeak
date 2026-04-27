# dpv-monopeak

Feature extraction for **Differential Pulse Voltammetry (DPV)** scans that contain a single dominant Faradaic peak.

```python
from dpv_monopeak import compute_dpv, extract_dpv_features
```

## Assumption

**This package assumes one and only one Faradaic peak per scan.** The baseline is estimated by a linear interpolation between the left and right current minima flanking that peak. If your scan contains multiple peaks, only the global maximum is processed.

## Installation

```
pip install dpv-monopeak
```

## Quick start: loading from a CSV file

If your instrument exports each scan as a CSV with voltage and current columns, this is the typical workflow:

```
v_V,i_uA
-0.400,-305.1
-0.390,-298.4
-0.380,-289.7
-0.370,-279.1
...
0.390,612.3
0.400,598.8
```

```python
import pandas as pd
import numpy as np
from dpv_monopeak import extract_dpv_features

df = pd.read_csv("scan.csv")

# rename to whatever your instrument calls the columns
v = df["v_V"].to_numpy(dtype=float)
i = df["i_uA"].to_numpy(dtype=float)

feats = extract_dpv_features(v, i)
if feats is not None:
    print(f"Peak voltage : {feats['X']:.4f} V")
    print(f"Peak height  : {feats['Height']:.2f} µA")
    print(f"Peak area    : {feats['Area']:.4f} µA·V")
    print(f"FWHM         : {feats['Width']:.4f} V")
else:
    print("No valid peak found in this scan.")
```

To process a folder of single-scan CSVs in one go:

```python
import pathlib, pandas as pd, numpy as np
from dpv_monopeak import extract_dpv_features

results = []
for path in sorted(pathlib.Path("scans/").glob("*.csv")):
    df = pd.read_csv(path)
    v = df["v_V"].to_numpy(dtype=float)
    i = df["i_uA"].to_numpy(dtype=float)
    feats = extract_dpv_features(v, i)
    if feats is not None:
        feats["file"] = path.name
        results.append(feats)

summary = pd.DataFrame(results)
print(summary[["file", "X", "Height", "Area", "Width"]])
```

## API

### `extract_dpv_features(v, i, left_chop_fn=_find_left_chop)`

| Parameter | Type | Description |
|---|---|---|
| `v` | `np.ndarray` | Voltage array, shape `(N,)`, in volts |
| `i` | `np.ndarray` | Current array, shape `(N,)`, same length as `v` |
| `left_chop_fn` | callable | Controls the left baseline anchor search — see [Left-chop heuristic](#left-chop-heuristic) below |

Returns a dict with the 8 electrochemical features, or `None` if no valid peak is found.

### `compute_dpv(v, i, left_chop_fn=_find_left_chop)`

| Parameter | Type | Description |
|---|---|---|
| `v` | `np.ndarray` | Voltage array, shape `(N,)`, in volts |
| `i` | `np.ndarray` | Current array, shape `(N,)`, same length as `v` |
| `left_chop_fn` | callable | Controls the left baseline anchor search — see [Left-chop heuristic](#left-chop-heuristic) below |

Returns the same 8 features **plus** all internal intermediates used to compute them. Useful when you need to plot the baseline, the corrected current, or the FWHM markers. Returns `None` if no valid peak is found.

### `plot_dpv(v, i, left_chop_fn=_find_left_chop, ax=None)`

| Parameter | Type | Description |
|---|---|---|
| `v` | `np.ndarray` | Voltage array, shape `(N,)`, in volts |
| `i` | `np.ndarray` | Current array, shape `(N,)`, same length as `v` |
| `left_chop_fn` | callable | Controls the left baseline anchor search — see [Left-chop heuristic](#left-chop-heuristic) below |
| `ax` | `tuple` or `None` | Pass a 2-tuple of existing Matplotlib `Axes` objects `(ax1, ax2)` to draw into them. If `None`, a new figure is created automatically. |

Returns `(ax1, ax2)` — the two Matplotlib axes used for the plot. Requires `pip install dpv-monopeak[plot]`.

---

## Feature reference

| Key | Units | Description |
|---|---|---|
| `X` | V | Peak voltage |
| `Height` | µA | Baseline-corrected peak current |
| `Area` | µA·V | Area under the corrected peak |
| `Width` | V | Full width at half maximum (FWHM) |
| `YOffset` | µA | Baseline current at peak voltage |
| `MaxSlope` | µA/V | Steepest slope on the rising edge |
| `MinSlope` | µA/V | Steepest slope on the falling edge (negative) |
| `SumSlope` | µA/V | `MaxSlope + \|MinSlope\|` |

### X - Peak voltage

The voltage at which the baseline-corrected current reaches its maximum. In DPV this corresponds closely to the formal redox potential (E°') of the electroactive species. It is the primary identifier of which analyte is being detected and shifts with surface chemistry changes, pH, and ionic strength.

### Height - Peak current

The peak current after the linear baseline has been subtracted (µA). This is the net Faradaic current arising from the redox reaction at the electrode surface. For a well-behaved system it scales with analyte concentration, making it the most direct quantitative feature. Units match whatever you pass in for `i`.

### Area - Peak area

The trapezoidal integral of the baseline-corrected current over the voltage window between the left and right baseline anchors (µA·V). It integrates over the full peak shape rather than just the tip, which makes it more robust to peak broadening or slight asymmetry. Proportional to total charge transferred during the pulse sequence.

### Width - Full width at half maximum

The voltage span between the two points on the corrected peak where the current equals exactly half the peak height (V). For an ideal reversible one-electron process in DPV the theoretical FWHM is approximately 90 mV at room temperature. Broader peaks indicate slower electron transfer kinetics, a wide distribution of binding-site energies (common in molecularly imprinted electrodes), or instrument noise at low concentrations.

### YOffset - Baseline at peak

The value of the estimated linear baseline evaluated at the peak voltage (µA). This is the non-Faradaic background current at that point in the sweep. Monitoring YOffset across a concentration series reveals electrode drift or fouling: a steadily changing YOffset with no change in analyte concentration indicates the background is shifting rather than the Faradaic signal.

### MaxSlope - Rising-edge slope

The maximum value of d(i\_corrected)/dV on the rising (anodic) side of the peak, between the left baseline anchor and the peak maximum (µA/V). A steeper rising edge indicates a sharper onset of the Faradaic reaction. Computed as a finite difference over the raw digitized data.

### MinSlope - Falling-edge slope

The minimum value of d(i\_corrected)/dV on the falling side of the peak, between the peak maximum and the right baseline anchor (µA/V). Negative by convention. A more negative value indicates a steeper decay after the peak. Computed as a finite difference over the raw digitized data.

### SumSlope - Combined sharpness

`MaxSlope + |MinSlope|`. A single number summarising overall peak sharpness: a high SumSlope means a narrow, well-defined peak; a low SumSlope means a broad or flat one. Because both slope features are derived from finite differences on raw data rather than smoothed curves, SumSlope values from this package should not be compared directly to slope values extracted by other software.

---

## Intermediate reference

`compute_dpv` returns these additional keys alongside the 8 features. They are intended for diagnostic plotting and are not needed for routine quantification.

| Key | Type | Description |
|---|---|---|
| `peak_idx` | int | Index of the global current maximum in `v` and `i` |
| `left_bound` | int | Index of the left baseline anchor |
| `right_bound` | int | Index of the right baseline anchor |
| `baseline` | array | Linear baseline current at every voltage point (µA) |
| `i_corrected` | array | Baseline-subtracted current: `i - baseline` (µA) |
| `half_max` | float | `Height / 2` — the threshold for FWHM detection (µA) |
| `left_hw_idx` | int | Index of the left half-maximum crossing in `v` |
| `right_hw_idx` | int | Index of the right half-maximum crossing in `v` |
| `di` | array | Numerical derivative d(i\_corrected)/dV at midpoints (µA/V) |
| `v_mid` | array | Voltage midpoints corresponding to each `di` value (V) |
| `max_slope_idx` | int | Index into `di` where `MaxSlope` occurs |
| `min_slope_idx` | int | Index into `di` where `MinSlope` occurs |

**`peak_idx`, `left_bound`, `right_bound`** — the three landmark indices. `left_bound` and `right_bound` are the current minima on either side of the peak; the linear baseline is drawn between `(v[left_bound], i[left_bound])` and `(v[right_bound], i[right_bound])`.

**`baseline` and `i_corrected`** — the corrected current is simply `i - baseline`. Plotting both `i` and `baseline` on the same axes (as `plot_dpv` does) shows whether the background subtraction looks physically reasonable for your scan.

**`half_max`, `left_hw_idx`, `right_hw_idx`** — the three quantities needed to draw the FWHM bracket. `Width = v[right_hw_idx] - v[left_hw_idx]`. If the peak is very close to the voltage window edge and one crossing cannot be found, `Width` is returned as `np.nan`.

**`di` and `v_mid`** — the derivative array has length `len(v) - 1` and is evaluated at the midpoints between adjacent voltage samples. Plot `di` against `v_mid` to see the slope profile of the corrected peak.

**`max_slope_idx` and `min_slope_idx`** — indices into `di` (not into `v`) locating where `MaxSlope` and `MinSlope` occur. Use `v_mid[max_slope_idx]` to get the voltage of the steepest rising point.

---

## Diagnostic plot *(optional)*

```
pip install dpv-monopeak[plot]
```

```python
from dpv_monopeak.plot import plot_dpv

ax1, ax2 = plot_dpv(v, i)
```

`plot_dpv` produces a two-panel figure: the top panel shows the raw current with the baseline overlaid and the anchor points and peak position marked; the bottom panel shows the baseline-corrected current with the integrated area shaded and the FWHM bracket drawn. This is the fastest way to check whether the baseline estimation is sensible for a given scan.

---

## Left-chop heuristic

DPV scans from the target instrument show a rising baseline hump to the left of the Faradaic peak (a capacitive artifact from the electrode double layer). Naively including this rising region causes the left anchor search to latch onto the hump rather than the true pre-peak baseline.

The default heuristic (`_find_left_chop`):

1. Locate the peak index `p`.
2. Examine the **quarter window** `i[0 : p // 2]`.
3. The index of the maximum in that window is treated as the shoulder of the capacitive hump.
4. The left anchor search begins from that index, excluding the hump.

### How to use a custom `left_chop_fn`

The `left_chop_fn` parameter is accepted by all three functions: `extract_dpv_features`, `compute_dpv`, and `plot_dpv`. It must have the signature:

```python
def my_chop(i: np.ndarray, peak_idx: int) -> int:
    ...
```

It receives the full current array and the index of the peak maximum. It must return an integer index. The left baseline anchor will then be found as the current minimum in `i[returned_index : peak_idx]`.

**Disable the heuristic entirely**: use the full scan for the left anchor search:

```python
from dpv_monopeak import extract_dpv_features

feats = extract_dpv_features(v, i, left_chop_fn=lambda i, p: 0)
```

**Skip a fixed number of points from the left edge**: e.g. always ignore the first 10 points regardless of peak position:

```python
feats = extract_dpv_features(v, i, left_chop_fn=lambda i, p: 10)
```

**Suppress everything below a fixed voltage** - e.g. ignore everything below -0.2 V:

```python
import numpy as np
from dpv_monopeak import extract_dpv_features

def chop_below_voltage(v_array, threshold):
    def fn(i, peak_idx):
        return int(np.searchsorted(v_array, threshold))
    return fn

feats = extract_dpv_features(v, i, left_chop_fn=chop_below_voltage(v, -0.2))
```

**Use the same custom function with the diagnostic plot.** All three functions accept the same callable, so you can pass it directly:

```python
from dpv_monopeak.plot import plot_dpv

my_chop = lambda i, p: 10
ax1, ax2 = plot_dpv(v, i, left_chop_fn=my_chop)
```

If your instrument does not produce the capacitive hump artifact, the default heuristic is a no-op: the quarter-window maximum will fall near index 0 and the left anchor search covers almost the full left half of the scan. You do not need to override `left_chop_fn` in that case.

---

## License

MIT
