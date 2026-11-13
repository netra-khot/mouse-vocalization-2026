import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import extract_ridges

TRACKERS = ("main", "bottom", "top")
THRESHOLD_KHZ, MIN_FRAMES, TRIM_MS, MIN_GOOD_FRAMES = 10, 2, 5, 10
BAND_EDGES_KHZ = np.arange(20, 105, 5)
INTEGER_TOL = 0.15


def detect_jumps(times, ridge, amp, active, threshold_khz=THRESHOLD_KHZ,
                 min_frames=MIN_FRAMES, trim_ms=TRIM_MS):
    good = np.isfinite(ridge) & active
    idx = np.flatnonzero(good)
    if len(idx) < MIN_GOOD_FRAMES:
        return None, None
    start_ms, end_ms = times[idx[0]] * 1000, times[idx[-1]] * 1000
    span = dict(start_ms=start_ms, end_ms=end_ms, good_frames=len(idx),
                occupancy=np.histogram(ridge[good] / 1000, bins=BAND_EDGES_KHZ)[0])

    amp_db = 20 * np.log10(amp + 1e-12)
    rows = []
    changes = np.diff(ridge) / 1000
    for j in np.flatnonzero(good[1:] & good[:-1] & (np.abs(changes) >= threshold_khz)) + 1:
        pre, post = ridge[max(0, j - min_frames):j], ridge[j:j + min_frames]
        if len(pre) < min_frames or len(post) < min_frames:
            continue
        if not (np.isfinite(pre).all() and np.isfinite(post).all()):
            continue
        pre_khz, post_khz = np.median(pre) / 1000, np.median(post) / 1000
        if abs(post_khz - pre_khz) < threshold_khz:
            continue
        t_ms = times[j] * 1000
        if t_ms - start_ms <= trim_ms or end_ms - t_ms <= trim_ms:
            continue
        rows.append(dict(frame=int(j), time_ms=t_ms, pre_khz=pre_khz, post_khz=post_khz,
                         position=(t_ms - start_ms) / (end_ms - start_ms),
                         pre_amp_db=np.nanmedian(amp_db[max(0, j - 4):j]),
                         post_amp_db=np.nanmedian(amp_db[j:j + 4])))
    return rows, span


def add_jump_columns(jumps):
    j = jumps.copy()
    j["delta_khz"] = j["post_khz"] - j["pre_khz"]
    j["size_khz"] = j["delta_khz"].abs()
    j["up"] = j["delta_khz"] > 0
    j["ratio"] = j["post_khz"] / j["pre_khz"]
    j["f_lower_khz"] = j[["pre_khz", "post_khz"]].min(axis=1)
    j["n"] = j["f_lower_khz"] / j["size_khz"]
    j["n_round"] = j["n"].round().astype(int)
    j["near_integer"] = (j["n"] - j["n_round"]).abs() <= INTEGER_TOL
    j["octave"] = ((j["ratio"] / 2 - 1).abs() < 0.05) | ((j["ratio"] / 0.5 - 1).abs() < 0.05)
    return j


def jump_stats(jumps, calls, name=""):
    n_calls = len(calls)
    s = {"calls": n_calls, "jumps": len(jumps)}
    if len(jumps) == 0:
        s["calls with no jumps (%)"] = 100.0
        return pd.Series(s, name=name)
    j = add_jump_columns(jumps)
    pos = j["position"]
    s.update({
        "jumps per call": round(len(j) / n_calls, 2),
        "calls with no jumps (%)": round(100 * (1 - j["file"].nunique() / n_calls), 1),
        "jumps per call (calls that have any)": round(len(j) / j["file"].nunique(), 2),
        "median size (kHz)": round(j["size_khz"].median(), 1),
        "size IQR (kHz)": f"{j['size_khz'].quantile(.25):.1f}-{j['size_khz'].quantile(.75):.1f}",
        "up (%)": round(100 * j["up"].mean(), 1),
        "octave jumps (%)": round(100 * j["octave"].mean(), 1),
        "median n = f_lower/|df|": round(j["n"].median(), 2),
        f"n within +/-{INTEGER_TOL} of integer (%)": round(100 * j["near_integer"].mean(), 1),
        "most common n (rounded)": int(j["n_round"].mode().iloc[0]),
        "most common n (near-integer only)": int(j.loc[j["near_integer"], "n_round"].mode().iloc[0])
        if j["near_integer"].any() else np.nan,
        "median position in call (0-1)": round(pos.median(), 2),
        "first / middle / last third (%)": "/".join(
            str(round(100 * x)) for x in np.histogram(pos, bins=[0, 1 / 3, 2 / 3, 1])[0] / len(pos)),
    })
    return pd.Series(s, name=name)


def rate_by_band(jumps, calls, frame_s, min_frames=20):
    occupancy = np.vstack(calls["occupancy"].to_numpy()).sum(axis=0)
    starts = np.histogram(jumps["pre_khz"], bins=BAND_EDGES_KHZ)[0] if len(jumps) else np.zeros(len(occupancy))
    rate = np.where(occupancy >= min_frames, starts / np.maximum(occupancy * frame_s, 1e-12), np.nan)
    return pd.DataFrame(dict(band_khz=BAND_EDGES_KHZ[:-1], seconds=occupancy * frame_s, jumps=starts,
                             jumps_per_s=rate))


def compare_trackers(wavs, n_calls=400, seed=2, threshold_khz=THRESHOLD_KHZ, save_csv="jumps_trackers.csv"):
    order = np.random.default_rng(seed).permutation(len(wavs))
    jump_rows, call_rows, n_used, frame_s = [], [], 0, None
    for i in order:
        if n_used >= n_calls:
            break
        path = str(wavs[int(i)])
        try:
            r = extract_ridges(path)
        except Exception:
            continue
        per_tracker = {name: detect_jumps(r["times"], r[name], r[name + "_amp"], r["active"],
                                          threshold_khz=threshold_khz) for name in TRACKERS}
        if per_tracker["main"][1] is None:
            continue
        n_used += 1
        frame_s = frame_s or float(np.median(np.diff(r["times"])))
        for name, (rows, span) in per_tracker.items():
            if span is None:
                continue
            call_rows.append(dict(file=path, tracker=name, **span))
            jump_rows += [dict(file=path, tracker=name, **row) for row in rows]

    calls = pd.DataFrame(call_rows)
    jumps = pd.DataFrame(jump_rows)
    if not jumps.empty:
        jumps = add_jump_columns(jumps)
    if save_csv:
        jumps.to_csv(save_csv, index=False)

    table = pd.concat([jump_stats(jumps[jumps["tracker"] == t] if len(jumps) else jumps,
                                  calls[calls["tracker"] == t], t) for t in TRACKERS], axis=1)
    rates = pd.concat({t: rate_by_band(jumps[jumps["tracker"] == t] if len(jumps) else jumps,
                                       calls[calls["tracker"] == t], frame_s).set_index("band_khz")
                       for t in TRACKERS}, axis=1)
    print(f"{n_used} calls; jumps: " + ", ".join(f"{t} {int((jumps['tracker'] == t).sum()) if len(jumps) else 0}"
                                                 for t in TRACKERS))
    return table, jumps, calls, rates


def plot(jumps, rates, save_as="jump_trackers.png"):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    size_bins = np.arange(0, 75, 2.5)
    n_bins = np.arange(0, 10.05, 0.1)
    for t, color in zip(TRACKERS, ("0.4", "C0", "C3")):
        j = jumps[jumps["tracker"] == t]
        ax[0].hist(j["size_khz"], bins=size_bins, histtype="step", lw=1.5, color=color, label=f"{t} ({len(j)})")
        ax[1].hist(j["n"].clip(upper=10), bins=n_bins, histtype="step", lw=1.5, color=color, label=t)
        ax[2].plot(rates.index + 2.5, rates[(t, "jumps_per_s")], "o-", color=color, label=t)
    for k in range(1, 11):
        ax[1].axvline(k, color="k", lw=0.5, ls=":")
    ax[0].set(xlabel="|jump| (kHz)", ylabel="jumps", title="Jump size")
    ax[1].set(xlabel="n = f_lower / |df|", ylabel="jumps", title="Mode-number estimate")
    ax[2].set(xlabel="starting frequency (kHz)", ylabel="jumps per second in band", title="Jump rate by band")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(save_as, dpi=150)
    plt.show()
