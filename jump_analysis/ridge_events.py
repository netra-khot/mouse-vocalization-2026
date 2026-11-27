import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

from utils import extract_ridges, extract_dual_ridges

HANDOFF_WINDOW = 2
HANDOFF_OVERLAP = 5
MIN_SEP_KHZ = 3
TRIM_MS = 5
BIG_JUMP_KHZ = 10
INTEGER_TOL = 0.15
BAND_EDGES_KHZ = np.arange(20, 105, 5)


def call_events(d, file="", handoff_window=HANDOFF_WINDOW, handoff_overlap=HANDOFF_OVERLAP,
                min_sep_khz=MIN_SEP_KHZ):
    ridges, times = d["ridges"], d["times"]
    if not ridges:
        return [], [], None
    start = [int(r["frames"][0]) for r in ridges]
    end = [int(r["frames"][-1]) for r in ridges]
    call_start, call_end = min(start), max(end)
    t0, t1 = times[call_start] * 1000, times[call_end] * 1000

    def row(kind, frame, **kw):
        t = times[frame] * 1000
        return dict(file=file, type=kind, frame=frame, time_ms=t,
                    position=(t - t0) / max(t1 - t0, 1e-9),
                    near_edge=(t - t0 <= TRIM_MS) or (t1 - t <= TRIM_MS), **kw)

    events, used_end, used_start, handoff_pairs = [], set(), set(), set()
    candidates = []
    for q in range(len(ridges)):
        for r in range(len(ridges)):
            gap = start[r] - end[q]
            if q == r or start[q] >= start[r] or end[q] >= end[r] or not -handoff_overlap <= gap <= handoff_window + 1:
                continue
            pre, post = ridges[q]["freq_hz"][-1] / 1000, ridges[r]["freq_hz"][0] / 1000
            if abs(post - pre) >= min_sep_khz:
                candidates.append((abs(gap), q, r, pre, post))
    for _, q, r, pre, post in sorted(candidates):
        if q in used_end or r in used_start:
            continue
        used_end.add(q); used_start.add(r); handoff_pairs.add(frozenset((q, r)))
        events.append(row("handoff", start[r], pre_khz=pre, post_khz=post,
                          pre_amp=ridges[q]["amplitude"][-1], post_amp=ridges[r]["amplitude"][0]))

    for i, r in enumerate(ridges):
        others = [k for k in range(len(ridges)) if k != i]
        if i not in used_start:
            present = any(start[k] < start[i] <= end[k] for k in others)
            events.append(row("appear" if present else "onset", start[i], freq_khz=r["freq_hz"][0] / 1000))
        if i not in used_end:
            present = any(start[k] <= end[i] < end[k] for k in others)
            events.append(row("disappear" if present else "offset", end[i], freq_khz=r["freq_hz"][-1] / 1000))

    pairs = []
    for a in range(len(ridges)):
        for b in range(a + 1, len(ridges)):
            common, ia, ib = np.intersect1d(ridges[a]["frames"], ridges[b]["frames"], return_indices=True)
            if len(common) == 0:
                continue
            fa, fb = ridges[a]["freq_hz"][ia], ridges[b]["freq_hz"][ib]
            lo, hi = np.minimum(fa, fb), np.maximum(fa, fb)
            amp_lo = np.where(fa <= fb, ridges[a]["amplitude"][ia], ridges[b]["amplitude"][ib])
            amp_hi = np.where(fa <= fb, ridges[b]["amplitude"][ib], ridges[a]["amplitude"][ia])
            pairs.append(dict(file=file, handoff=frozenset((a, b)) in handoff_pairs, frames=len(common), duration_ms=len(common) * (times[1] - times[0]) * 1000,
                              f_low_khz=np.median(lo) / 1000, f_high_khz=np.median(hi) / 1000,
                              ratio=np.median(hi / lo), n=np.median(lo / (hi - lo)),
                              amp_high_over_low=np.median(amp_hi / amp_lo)))

    frames_any = np.zeros(len(times), dtype=int)
    for r in ridges:
        frames_any[r["frames"]] += 1
    occupancy = np.histogram(np.concatenate([r["freq_hz"] for r in ridges]) / 1000, bins=BAND_EDGES_KHZ)[0]
    call = dict(file=file, ridges=len(ridges), ridge_frames=int((frames_any > 0).sum()),
                multi_frames=int((frames_any > 1).sum()), occupancy=occupancy)
    return events, pairs, call


def add_handoff_columns(h):
    h = h.copy()
    h["delta_khz"] = h["post_khz"] - h["pre_khz"]
    h["size_khz"] = h["delta_khz"].abs()
    h["up"] = h["delta_khz"] > 0
    h["ratio"] = h["post_khz"] / h["pre_khz"]
    h["n"] = h[["pre_khz", "post_khz"]].min(axis=1) / h["size_khz"]
    h["big"] = h["size_khz"] >= BIG_JUMP_KHZ
    return h


def _n_summary(n, prefix):
    if len(n) == 0:
        return {}
    near = (n - n.round()).abs() <= INTEGER_TOL
    return {f"{prefix}: median n": round(n.median(), 2),
            f"{prefix}: n within +/-{INTEGER_TOL} of integer (%)": round(100 * near.mean(), 1),
            f"{prefix}: most common n (near-integer only)": int(n[near].round().mode().iloc[0]) if near.any() else np.nan}


def dual_stats(events, pairs, calls, main_jumps_per_call, name=""):
    n_calls = len(calls)
    ev = events[~events["near_edge"]]
    if len(pairs):
        pairs = pairs[~pairs["handoff"]]
    h = add_handoff_columns(ev[ev["type"] == "handoff"])
    s = {
        "calls": n_calls,
        "ridges per call (median)": calls["ridges"].median(),
        "calls with 2+ ridges at once (%)": round(100 * (calls["multi_frames"] > 0).mean(), 1),
        "frames with 2+ ridges (% of ridge frames)": round(100 * calls["multi_frames"].sum() / calls["ridge_frames"].sum(), 1),
        "coexisting pairs": len(pairs),
        "coexisting pair duration, median (ms)": round(pairs["duration_ms"].median(), 1) if len(pairs) else np.nan,
        "handoffs per call": round(len(h) / n_calls, 2),
        "handoffs >= 10 kHz per call": round(h["big"].sum() / n_calls, 2),
        "main-tracker jumps per call (same calls)": round(main_jumps_per_call, 2),
        "handoffs < 10 kHz per call": round((~h["big"]).sum() / n_calls, 2),
        "appear (while another ridge continues) per call": round((ev["type"] == "appear").sum() / n_calls, 2),
        "disappear (while another ridge continues) per call": round((ev["type"] == "disappear").sum() / n_calls, 2),
        "handoffs up (%)": round(100 * h["up"].mean(), 1) if len(h) else np.nan,
        "handoff size, median (kHz)": round(h["size_khz"].median(), 1) if len(h) else np.nan,
        "handoff ratio after/before, median (>=10 kHz, up)": round(h.loc[h["big"] & h["up"], "ratio"].median(), 2),
        "pair ratio f_high/f_low, median": round(pairs["ratio"].median(), 2) if len(pairs) else np.nan,
    }
    if len(pairs):
        s.update(_n_summary(pairs["n"], "coexisting pairs"))
    s.update(_n_summary(h.loc[h["big"], "n"], "handoffs >= 10 kHz"))
    s.update(_n_summary(h.loc[~h["big"], "n"], "handoffs < 10 kHz"))
    return pd.Series(s, name=name), h


def handoff_rate_by_band(h, calls, frame_s, min_frames=20):
    occupancy = np.vstack(calls["occupancy"].to_numpy()).sum(axis=0)
    starts = np.histogram(h["pre_khz"], bins=BAND_EDGES_KHZ)[0]
    rate = np.where(occupancy >= min_frames, starts / np.maximum(occupancy * frame_s, 1e-12), np.nan)
    return pd.DataFrame(dict(band_khz=BAND_EDGES_KHZ[:-1], seconds=occupancy * frame_s, handoffs=starts,
                             handoffs_per_s=rate)).set_index("band_khz")


def run(wavs, n_calls=400, seed=2, min_frames_settings=(3, 5), save_prefix="dual_ridge"):
    # same call selection as jump_stats.compare_trackers: seed-2 order, main tracker >= 10 good frames
    from jump_stats import detect_jumps
    order = np.random.default_rng(seed).permutation(len(wavs))
    files, main_jumps, frame_s = [], 0, None
    results = {m: ([], [], []) for m in min_frames_settings}
    for i in order:
        if len(files) >= n_calls:
            break
        path = str(wavs[int(i)])
        try:
            r = extract_ridges(path)
            rows, span = detect_jumps(r["times"], r["main"], r["main_amp"], r["active"])
            if span is None:
                continue
            duals = {m: extract_dual_ridges(path, min_frames=m) for m in min_frames_settings}
        except Exception:
            continue
        files.append(path)
        main_jumps += len(rows)
        frame_s = frame_s or float(np.median(np.diff(r["times"])))
        for m, d in duals.items():
            ev, pr, call = call_events(d, path)
            if call is None:
                call = dict(file=path, ridges=0, ridge_frames=0, multi_frames=0,
                            occupancy=np.zeros(len(BAND_EDGES_KHZ) - 1, dtype=int))
            results[m][0].extend(ev); results[m][1].extend(pr); results[m][2].append(call)

    tables, rates, out = [], {}, {}
    for m, (ev, pr, cl) in results.items():
        events, pairs, calls = pd.DataFrame(ev), pd.DataFrame(pr), pd.DataFrame(cl)
        name = f"min ridge {m} frames"
        table, h = dual_stats(events, pairs, calls, main_jumps / len(files), name)
        tables.append(table)
        rates[name] = handoff_rate_by_band(h, calls, frame_s)
        out[name] = dict(events=events, pairs=pairs, calls=calls, handoffs=h)
        if save_prefix:
            events.to_csv(f"{save_prefix}_events_min{m}.csv", index=False)
            pairs.to_csv(f"{save_prefix}_pairs_min{m}.csv", index=False)
    table = pd.concat(tables, axis=1)
    rates = pd.concat(rates, axis=1)
    if save_prefix:
        table.to_csv(f"{save_prefix}_summary.csv")
        rates.to_csv(f"{save_prefix}_rates.csv")
        pd.Series(files, name="file").to_csv(f"{save_prefix}_calls.csv", index=False)
    print(f"{len(files)} calls")
    return table, rates, out


def plot(out, rates, name, save_as="dual_ridge.png"):
    o = out[name]
    h, pairs = o["handoffs"], o["pairs"]
    pairs = pairs[~pairs["handoff"]] if len(pairs) else pairs
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.5))
    ax[0].hist(h["size_khz"], bins=np.arange(0, 60, 1.5), color="C0")
    ax[0].axvline(BIG_JUMP_KHZ, color="k", ls="--", lw=0.8)
    ax[0].set(xlabel="|handoff| (kHz)", ylabel="handoffs", title="Handoff size (dashed = old 10 kHz cutoff)")
    n_bins = np.arange(0, 10.05, 0.2)
    ax[1].hist(pairs["n"][pairs["n"] <= 10], bins=n_bins, histtype="step", lw=1.5,
               label=f"coexisting pairs ({(pairs['n'] <= 10).sum()} of {len(pairs)})")
    ax[1].hist(h.loc[h["big"], "n"], bins=n_bins, histtype="step", lw=1.5,
               label=f"handoffs >= {BIG_JUMP_KHZ} kHz ({h['big'].sum()})")
    for k in range(1, 11):
        ax[1].axvline(k, color="k", lw=0.4, ls=":")
    ax[1].set(xlabel="n = f_low / (f_high - f_low)", ylabel="count", title="Mode-number estimate (n <= 10 shown)")
    r_bins = np.linspace(1, 2.4, 57)
    ax[2].hist(pairs["ratio"], bins=r_bins, histtype="step", lw=1.5, label="coexisting pairs f_high/f_low")
    big_ratio = np.maximum(h["ratio"], 1 / h["ratio"])
    ax[2].hist(big_ratio, bins=r_bins, histtype="step", lw=1.5, label="handoffs (larger/smaller)")
    top = ax[2].get_ylim()[1]
    for k, (v, txt) in enumerate([(5 / 4, "5/4"), (4 / 3, "4/3"), (3 / 2, "3/2"), (5 / 3, "5/3"), (2, "2")]):
        ax[2].axvline(v, color="k", lw=0.6, ls=":")
        ax[2].text(v, top * (0.95 if k % 2 == 0 else 0.87), txt, ha="center", fontsize=8)
    ax[2].set(xlabel="frequency ratio", title="Ratios between ridges")
    ax[3].plot(rates.index + 2.5, rates[(name, "handoffs_per_s")], "o-")
    ax[3].set(xlabel="starting frequency (kHz)", ylabel="handoffs per second in band", title="Handoff rate by band")
    for a in ax[1:3]:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_as, dpi=150)
    plt.show()


def plot_calls(paths, save_as="dual_ridge_examples.png", **tracker_kwargs):
    cols = 4
    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
    for a, p in zip(axes.ravel(), paths):
        d = extract_dual_ridges(p, **tracker_kwargs)
        ev, _, _ = call_events(d, str(p))
        a.pcolormesh(d["times"] * 1000, d["freqs"] / 1000, librosa.amplitude_to_db(d["mag"], ref=np.max),
                     shading="auto", cmap="gray_r", vmin=-60, vmax=0)
        for k, r in enumerate(d["ridges"]):
            a.plot(r["time_s"] * 1000, r["freq_hz"] / 1000, ".-", ms=3, lw=1.2, color=f"C{k % 10}")
        marks = {"handoff": ("r", "v"), "appear": ("lime", "^"), "disappear": ("orange", "x")}
        for e in ev:
            if e["type"] in marks and not e["near_edge"]:
                y = e.get("post_khz", e.get("freq_khz"))
                a.plot(e["time_ms"], y, marks[e["type"]][1], color=marks[e["type"]][0], ms=8, mew=2)
        if d["ridges"]:
            t = np.concatenate([r["time_s"] for r in d["ridges"]]) * 1000
            a.set_xlim(t.min() - 5, t.max() + 5)
        a.set_ylim(20, 100)
        a.set_title(f"{p.stem if hasattr(p, 'stem') else p}: {len(d['ridges'])} ridges", fontsize=9)
    for a in axes.ravel()[len(paths):]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(save_as, dpi=120)
    plt.show()
