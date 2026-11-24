import pickle
from pathlib import Path

import numpy as np

from utils import extract_ridges


def remove_jumps(f, threshold):
    # adviser's rule: zero out steps bigger than threshold, rebuild by cumulative sum from f[0]
    df = np.diff(f)
    is_jump = np.abs(df) > threshold
    df[is_jump] = 0
    f_clean = f[0] + np.concatenate([[0], np.cumsum(df)])
    return f_clean, np.where(is_jump)[0] + 1   # cleaned trajectory, jump indices


def remove_jumps_track(track, threshold, anchor="first"):
    # tracks have NaN where the call is inactive or a short piece was dropped; run remove_jumps on
    # the finite frames only (a step across a NaN gap counts as one step) and keep NaNs in place
    # anchor="first": f_clean starts at f[0] (adviser's version)
    # anchor="longest": shift f_clean by a constant so it matches the original on the longest
    #                   jump-free stretch (same shape, but it sits on the dominant ridge)
    f_clean = np.full(len(track), np.nan)
    good = np.flatnonzero(np.isfinite(track))
    if len(good) == 0:
        return f_clean, np.array([], dtype=int)
    cleaned, jumps = remove_jumps(track[good], threshold)
    if anchor == "longest" and len(jumps):
        edges = np.concatenate([[0], jumps, [len(good)]])
        i = np.argmax(np.diff(edges))
        seg = slice(edges[i], edges[i + 1])
        cleaned = cleaned + np.median(track[good][seg] - cleaned[seg])
    f_clean[good] = cleaned
    return f_clean, good[jumps]   # jump indices are frame indices in the original track


# ---------------------------------------------------------------------------------------------
# FROZEN copies: a snapshot of remove_jumps / remove_jumps_track as of 2026-11-23, kept for a
# collaborator who needs this exact behaviour. Do NOT edit these; change the originals above instead.
# ---------------------------------------------------------------------------------------------
def remove_jumps_frozen(f, threshold):
    df = np.diff(f)
    is_jump = np.abs(df) > threshold
    df[is_jump] = 0
    f_clean = f[0] + np.concatenate([[0], np.cumsum(df)])
    return f_clean, np.where(is_jump)[0] + 1


def remove_jumps_track_frozen(track, threshold, anchor="first"):
    f_clean = np.full(len(track), np.nan)
    good = np.flatnonzero(np.isfinite(track))
    if len(good) == 0:
        return f_clean, np.array([], dtype=int)
    cleaned, jumps = remove_jumps_frozen(track[good], threshold)
    if anchor == "longest" and len(jumps):
        edges = np.concatenate([[0], jumps, [len(good)]])
        i = np.argmax(np.diff(edges))
        seg = slice(edges[i], edges[i + 1])
        cleaned = cleaned + np.median(track[good][seg] - cleaned[seg])
    f_clean[good] = cleaned
    return f_clean, good[jumps]


def extract_tracks(wavs, tracker="main", min_good_frames=10, save_as="tracks_main.pkl", **ridge_kwargs):
    # slow part (reads audio): one track per call, same time base as extract_ridges
    tracks = {}
    for k, path in enumerate(wavs):
        try:
            r = extract_ridges(str(path), **ridge_kwargs)
        except Exception:
            continue
        if np.isfinite(r[tracker]).sum() < min_good_frames:
            continue
        tracks[Path(path).stem] = dict(file=str(path), time_s=r["times"], frequency_hz=r[tracker],
                                       amplitude=r[tracker + "_amp"])
        if (k + 1) % 1000 == 0:
            print(f"{k + 1}/{len(wavs)} files, {len(tracks)} calls kept")
    if save_as:
        with open(save_as, "wb") as fh:
            pickle.dump(tracks, fh)
    return tracks


def all_steps(tracks):
    # |df| for every step between consecutive finite frames, all calls pooled (Hz)
    return np.concatenate([np.abs(np.diff(t["frequency_hz"][np.isfinite(t["frequency_hz"])]))
                           for t in tracks.values()])


def clean_all(tracks, threshold, save_clean="tracks_clean.pkl", save_jumps="jump_indices.pkl"):
    clean, jumps = {}, {}
    for key, t in tracks.items():
        f_clean, idx = remove_jumps_track(t["frequency_hz"], threshold)
        clean[key] = dict(time_s=t["time_s"], frequency_hz=f_clean)
        jumps[key] = idx
    if save_clean:
        with open(save_clean, "wb") as fh:
            pickle.dump(clean, fh)
    if save_jumps:
        with open(save_jumps, "wb") as fh:
            pickle.dump(jumps, fh)
    return clean, jumps
