# Task: remove frequency jumps from the tracked trajectories

## Background (keep brief)
This is a mouse USV project. The tracker extracts the main frequency trajectory f[k] (one value per time step) from each call's spectrogram. Some calls contain sudden large jumps, which are mode changes. We want to ignore them without trying to infer mode numbers.

## What to do (adviser's suggestion, exactly)
Given the frequencies f[k] at steps k:
1. Compute `df[k] = f[k] - f[k-1]`.
2. Set `df[k] = 0` wherever `|df[k]| > threshold`.
3. Rebuild the trajectory from the cleaned steps by cumulative sum, starting at f[0].

```python
import numpy as np

def remove_jumps(f, threshold):
    df = np.diff(f)
    is_jump = np.abs(df) > threshold
    df[is_jump] = 0
    f_clean = f[0] + np.concatenate([[0], np.cumsum(df)])
    return f_clean, np.where(is_jump)[0] + 1   # cleaned trajectory, jump indices
```

## Choosing the threshold
- Plot a histogram of |df| across all calls. Normal step-to-step changes should be small; jumps are large (median jump ≈ 30 kHz in earlier analysis). Put the threshold in the gap between the two.
- Report the % of steps removed at the chosen threshold.

## Output
- `f_clean` for every call, same length and time base as the original track.
- Jump indices for every call, saved separately.
- About 10 plots showing the spectrogram, the original track, and `f_clean` overlaid, to check that it worked.

## Steps
1. Find where the tracker's per-call frequency output lives in this repo.
2. Add `remove_jumps` and run it on all calls.
3. Make the |df| histogram, pick the threshold, and make the check plots.