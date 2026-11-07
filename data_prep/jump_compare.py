from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
SETTINGS = {
    "old tracker": dict(MIN_SNR_DB=None, JUMP_PENALTY=0.03),
    "new tracker": dict(MIN_SNR_DB=15, JUMP_PENALTY=2),
}
THRESHOLD_KHZ, MIN_FRAMES, TRIM_MS, N_CALLS = 10, 2, 5, 400
 
 
def _jumps_in_call(path, ja):
    """Same detection as jump_analysis.inspect_file, plus loudness either side. No images."""
    times, freqs, mag, ridge, active = ja.extract_raw_ridge(Path(path))
    good = np.isfinite(ridge) & active
    if good.sum() < 10:
        return None                                   # too little signal to count as a call
    idx = np.flatnonzero(good)
    start_ms, end_ms = times[idx[0]] * 1000, times[idx[-1]] * 1000
 
    amp_db = np.full(len(ridge), np.nan)              # loudness of the tracked band at each frame
    k = np.abs(freqs[None, :] - ridge[idx][:, None]).argmin(axis=1)
    amp_db[idx] = 20 * np.log10(mag[k, idx] + 1e-12)
 
    rows = []
    changes = np.diff(ridge) / 1000
    for j in np.flatnonzero(good[1:] & good[:-1] & (np.abs(changes) >= THRESHOLD_KHZ)) + 1:
        pre, post = ridge[max(0, j - MIN_FRAMES):j], ridge[j:j + MIN_FRAMES]
        if len(pre) < MIN_FRAMES or len(post) < MIN_FRAMES:
            continue
        if not (np.isfinite(pre).all() and np.isfinite(post).all()):
            continue
        pre_khz, post_khz = np.median(pre) / 1000, np.median(post) / 1000
        if abs(post_khz - pre_khz) < THRESHOLD_KHZ:
            continue
        t_ms = times[j] * 1000
        if t_ms - start_ms <= TRIM_MS or end_ms - t_ms <= TRIM_MS:
            continue
        rows.append(dict(file=str(path), time_ms=t_ms, pre_khz=pre_khz, post_khz=post_khz,
                         amp_change_db=np.nanmedian(amp_db[j:j + 4]) - np.nanmedian(amp_db[max(0, j - 4):j])))
    return rows
 
 
def _summary(name, jumps, n_calls):
    s = {"calls": n_calls}
    if len(jumps) == 0:
        s.update({"jumps": 0, "calls with no jumps (%)": 100.0})
        return pd.Series(s, name=name)
    j = jumps.sort_values(["file", "time_ms"]).copy()
    j["ratio"] = j["post_khz"] / j["pre_khz"]
    j["up"] = j["ratio"] > 1
    octave = (np.abs(j["ratio"] / 2 - 1) < 0.05) | (np.abs(j["ratio"] / 0.5 - 1) < 0.05)
 
    # what happens next in the same call
    nxt = j.groupby("file").shift(-1)
    has_next = nxt["time_ms"].notna()
    reverses = has_next & (nxt["up"] != j["up"])
    returns = reverses & (np.abs(nxt["post_khz"] / j["pre_khz"] - 1) < 0.1)
 
    s.update({
        "jumps": len(j),
        "calls with no jumps (%)": round(100 * (1 - j["file"].nunique() / n_calls), 1),
        "jumps per call (calls that have any)": round(len(j) / j["file"].nunique(), 2),
        "octave jumps, 2x or 0.5x (%)": round(100 * octave.mean(), 1),
        "up / down": f"{int(j['up'].sum())} / {int((~j['up']).sum())}",
        "median ratio, up jumps": round(j.loc[j["up"], "ratio"].median(), 2),
        "median ratio, down jumps": round(j.loc[~j["up"], "ratio"].median(), 2),
        "median size (kHz)": round((j["post_khz"] - j["pre_khz"]).abs().median(), 1),
        "followed by another jump (%)": round(100 * has_next.mean(), 1),
        "...that goes back the other way (%)": round(100 * reverses[has_next].mean(), 1) if has_next.any() else np.nan,
        "hop and return within 10% of start (%)": round(100 * returns.mean(), 1),
        "median time to next jump (ms)": round((nxt["time_ms"] - j["time_ms"])[has_next].median(), 1) if has_next.any() else np.nan,
        "median loudness change across jump (dB)": round(j["amp_change_db"].median(), 1),
    })
    return pd.Series(s, name=name)
 
 
def compare(wavs, ja, n_calls=N_CALLS, seed=2):
    """Run old and new tracker on the same calls. Returns (summary table, all jumps)."""
    saved = {k: getattr(ja, k) for k in ("MIN_SNR_DB", "JUMP_PENALTY")}
    order = np.random.default_rng(seed).permutation(len(wavs))   # same order as cell 9
    results, calls = {}, None
    try:
        for name, setting in SETTINGS.items():
            for k, v in setting.items():
                setattr(ja, k, v)
            rows, used = [], []
            candidates = order if calls is None else calls
            for i in candidates:
                if calls is None and len(used) >= n_calls:
                    break
                try:
                    r = _jumps_in_call(wavs[int(i)], ja)
                except Exception:
                    continue
                if r is None:
                    if calls is not None:
                        used.append(i)     # same call as the old run, just no usable track now
                    continue
                used.append(i)
                rows += r
            if calls is None:
                calls = used               # the new tracker reuses exactly these calls
            df = pd.DataFrame(rows)
            df["tracker"] = name
            results[name] = (df, len(calls))
            print(f"{name}: {len(calls)} calls, {len(df)} jumps")
    finally:
        for k, v in saved.items():
            setattr(ja, k, v)
 
    table = pd.concat([_summary(n, df, nc) for n, (df, nc) in results.items()], axis=1)
    jumps = pd.concat([df for df, _ in results.values()], ignore_index=True)
    return table, jumps
 
 
def plot(jumps, save_as="jump_compare.png"):
    """Left: jump ratios, old vs new, with simple whole-number ratios marked
    (if the jumps are between whistle modes n -> m, the ratio should be m/n).
    Right: direction and size vs. starting frequency for the new tracker."""
    jumps = jumps.assign(ratio=jumps["post_khz"] / jumps["pre_khz"])
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    bins = np.linspace(0.3, 2.4, 64)
    for name, color in [("old tracker", "0.6"), ("new tracker", "C0")]:
        r = jumps.loc[jumps["tracker"] == name, "ratio"]
        ax[0].hist(r, bins=bins, color=color, alpha=0.7, label=f"{name} ({len(r)} jumps)")
    top = ax[0].get_ylim()[1]
    marks = [(1/2, "1/2"), (3/5, "3/5"), (2/3, "2/3"), (3/2, "3/2"), (5/3, "5/3"), (2, "2")]
    for n, (value, text) in enumerate(marks):
        ax[0].axvline(value, color="k", lw=0.6, ls=":")
        ax[0].text(value, top * (0.97 if n % 2 == 0 else 0.89), text, ha="center", fontsize=8)
    ax[0].set_xlabel("frequency after jump / before"); ax[0].set_ylabel("number of jumps")
    ax[0].set_title("Jump ratios, old vs new tracker"); ax[0].legend(loc="center right")
 
    new = jumps[jumps["tracker"] == "new tracker"]
    up = new["ratio"] > 1
    ax[1].scatter(new.loc[up, "pre_khz"], (new["post_khz"] - new["pre_khz"])[up], s=10, label="up")
    ax[1].scatter(new.loc[~up, "pre_khz"], (new["post_khz"] - new["pre_khz"])[~up], s=10, label="down")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].legend()
    ax[1].set_xlabel("frequency before jump (kHz)"); ax[1].set_ylabel("jump (kHz)")
    ax[1].set_title("New tracker: direction and size vs. starting frequency")
    fig.tight_layout(); fig.savefig(save_as, dpi=150); plt.show()