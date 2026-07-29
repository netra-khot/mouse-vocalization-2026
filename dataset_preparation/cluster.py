from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from utils import get_main_freq_traj
except ImportError as exc:
    raise SystemExit(
        "Could not import get_main_freq_traj from utils.py. "
        "Put this script next to utils.py or fix the import."
    ) from exc

AUDIO_EXTENSIONS = {".wav", ".flac"}
CACHE_FILENAME = "contour_cache.npz"
ASSIGNMENTS_FILENAME = "cluster_assignments.csv"
LABEL_MAP_FILENAME = "cluster_labels.json"


@dataclass
class ContourRecord:
    source_audio: str
    call_index: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    raw_time_s: np.ndarray
    raw_frequency_hz: np.ndarray
    resampled_frequency_hz: np.ndarray
    feature_vector: np.ndarray


def find_audio_files(audio_dir: Path) -> list[Path]:
    return sorted(
        p for p in audio_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def contiguous_active_segments(
    active: np.ndarray,
    max_gap_bins: int = 2,
    min_active_bins: int = 5,
) -> list[tuple[int, int]]:
    active = np.asarray(active, dtype=bool).reshape(-1).copy()

    # Bridge short inactive gaps inside an otherwise active call.
    if max_gap_bins > 0 and active.size:
        i = 0
        while i < len(active):
            if active[i]:
                i += 1
                continue
            start = i
            while i < len(active) and not active[i]:
                i += 1
            end = i
            if (
                end - start <= max_gap_bins
                and start > 0
                and end < len(active)
                and active[start - 1]
                and active[end]
            ):
                active[start:end] = True

    padded = np.pad(active.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start >= min_active_bins
    ]


def clean_contour(
    times: Sequence[float],
    frequencies: Sequence[float],
) -> tuple[np.ndarray, np.ndarray] | None:
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    f = np.asarray(frequencies, dtype=np.float64).reshape(-1)

    valid = np.isfinite(t) & np.isfinite(f) & (f > 0)
    t = t[valid]
    f = f[valid]

    if len(f) < 5:
        return None

    order = np.argsort(t)
    t = t[order]
    f = f[order]

    t_unique, indices = np.unique(t, return_index=True)
    t = t_unique
    f = f[indices]

    if len(t) < 5 or t[-1] <= t[0]:
        return None

    return t, f


def resample_contour(
    times: np.ndarray,
    frequencies: np.ndarray,
    n_points: int,
) -> np.ndarray:
    normalized_time = (times - times[0]) / (times[-1] - times[0])
    target_time = np.linspace(0.0, 1.0, n_points)
    resampled = np.interp(target_time, normalized_time, frequencies)

    if n_points >= 9:
        window = min(9, n_points if n_points % 2 else n_points - 1)
        if window >= 5:
            resampled = savgol_filter(
                resampled,
                window_length=window,
                polyorder=2,
                mode="interp",
            )

    return resampled.astype(np.float32)


def count_significant_turning_points(
    frequencies_hz: np.ndarray,
    prominence_hz: float = 2_500.0,
) -> int:
    """Count meaningful peaks and valleys, ignoring small contour noise."""
    values = np.asarray(frequencies_hz, dtype=np.float64)
    if values.size < 5:
        return 0

    # A turning point must rise or fall by at least prominence_hz relative
    # to its local surroundings. This avoids counting tiny MFT fluctuations.
    peaks, _ = find_peaks(values, prominence=prominence_hz)
    valleys, _ = find_peaks(-values, prominence=prominence_hz)
    return int(len(peaks) + len(valleys))


def count_frequency_jumps(
    frequencies_hz: np.ndarray,
    threshold_hz: float = 5_000.0,
) -> int:
    """Count abrupt adjacent-bin frequency changes characteristic of steps."""
    return int(np.sum(np.abs(np.diff(frequencies_hz)) >= threshold_hz))


def make_feature_vector(
    resampled_hz: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    median_hz = float(np.median(resampled_hz))

    # Overall contour shape, independent of absolute pitch.
    relative_shape_khz = (
        resampled_hz - median_hz
    ) / 1000.0

    # Use endpoint medians to reduce sensitivity to noisy first/last bins.
    edge_points = max(3, len(resampled_hz) // 10)

    start_hz = float(
        np.median(resampled_hz[:edge_points])
    )
    end_hz = float(
        np.median(resampled_hz[-edge_points:])
    )

    frequency_range_khz = (
        float(np.max(resampled_hz))
        - float(np.min(resampled_hz))
    ) / 1000.0

    net_change_khz = (
        end_hz - start_hz
    ) / 1000.0

    slope_khz_per_s = (
        net_change_khz
        / max(duration_s, 1e-6)
    )

    turning_points = count_significant_turning_points(resampled_hz)

    step_count = count_frequency_jumps(
        resampled_hz
    )

    summary = np.array(
        [
            duration_s,
            frequency_range_khz,
            float(turning_points),
            float(step_count),
            net_change_khz,
            slope_khz_per_s,
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            relative_shape_khz.astype(np.float32),
            summary,
        ]
    )
def process_audio_file(
    audio_path: str,
    n_points: int,
    max_gap_bins: int,
    min_active_bins: int,
    min_duration_s: float,
    max_duration_s: float,
) -> tuple[list[ContourRecord], str | None]:
    path = Path(audio_path)

    try:
        times, freq_traj, active_bins = get_main_freq_traj(str(path))

        times = np.asarray(times, dtype=np.float64).reshape(-1)
        frequencies = np.asarray(freq_traj, dtype=np.float64).reshape(-1)
        active = np.asarray(active_bins, dtype=bool).reshape(-1)

        if not (len(times) == len(frequencies) == len(active)):
            raise ValueError(
                "get_main_freq_traj returned arrays of different lengths: "
                f"{len(times)}, {len(frequencies)}, {len(active)}"
            )

        segments = contiguous_active_segments(
            active,
            max_gap_bins=max_gap_bins,
            min_active_bins=min_active_bins,
        )

        records: list[ContourRecord] = []

        for call_index, (start, end) in enumerate(segments, start=1):
            cleaned = clean_contour(times[start:end], frequencies[start:end])
            if cleaned is None:
                continue

            segment_times, segment_frequencies = cleaned
            duration_s = float(segment_times[-1] - segment_times[0])

            if duration_s < min_duration_s or duration_s > max_duration_s:
                continue

            resampled = resample_contour(
                segment_times,
                segment_frequencies,
                n_points=n_points,
            )
            features = make_feature_vector(resampled, duration_s)

            records.append(
                ContourRecord(
                    source_audio=str(path.resolve()),
                    call_index=call_index,
                    start_time_s=float(segment_times[0]),
                    end_time_s=float(segment_times[-1]),
                    duration_s=duration_s,
                    raw_time_s=segment_times.astype(np.float32),
                    raw_frequency_hz=segment_frequencies.astype(np.float32),
                    resampled_frequency_hz=resampled,
                    feature_vector=features.astype(np.float32),
                )
            )

        return records, None

    except Exception as exc:
        return [], (
            f"{path}: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc(limit=2)}"
        )


def save_cache(records: list[ContourRecord], output_dir: Path) -> Path:
    raw_times = np.empty(len(records), dtype=object)
    raw_frequencies = np.empty(len(records), dtype=object)

    for i, record in enumerate(records):
        raw_times[i] = record.raw_time_s
        raw_frequencies[i] = record.raw_frequency_hz

    cache_path = output_dir / CACHE_FILENAME
    np.savez_compressed(
        cache_path,
        source_audio=np.asarray([r.source_audio for r in records], dtype=str),
        call_index=np.asarray([r.call_index for r in records], dtype=np.int32),
        start_time_s=np.asarray([r.start_time_s for r in records], dtype=np.float32),
        end_time_s=np.asarray([r.end_time_s for r in records], dtype=np.float32),
        duration_s=np.asarray([r.duration_s for r in records], dtype=np.float32),
        raw_time_s=raw_times,
        raw_frequency_hz=raw_frequencies,
        resampled_frequency_hz=np.stack(
            [r.resampled_frequency_hz for r in records]
        ).astype(np.float32),
        features=np.stack([r.feature_vector for r in records]).astype(np.float32),
    )
    return cache_path


def load_cache(output_dir: Path) -> dict[str, np.ndarray]:
    cache_path = output_dir / CACHE_FILENAME
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} does not exist. Run extract first."
        )
    with np.load(cache_path, allow_pickle=True) as cache:
        return {key: cache[key] for key in cache.files}


def command_extract(args: argparse.Namespace) -> None:
    audio_dir = Path(args.audio_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        raise SystemExit(f"No WAV or FLAC files found under {audio_dir}")

    if args.limit:
        audio_files = audio_files[:args.limit]

    print(f"Found {len(audio_files):,} audio files.")
    print(f"Using {args.jobs} worker(s).")

    worker_args = dict(
        n_points=args.points,
        max_gap_bins=args.max_gap_bins,
        min_active_bins=args.min_active_bins,
        min_duration_s=args.min_duration,
        max_duration_s=args.max_duration,
    )

    if args.jobs == 1:
        results = [
            process_audio_file(str(path), **worker_args)
            for path in tqdm(audio_files, desc="Extracting")
        ]
    else:
        results = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(process_audio_file)(str(path), **worker_args)
            for path in tqdm(audio_files, desc="Submitting")
        )

    records: list[ContourRecord] = []
    errors: list[str] = []
    for file_records, error in results:
        records.extend(file_records)
        if error:
            errors.append(error)

    if not records:
        raise SystemExit(
            "No usable contours were extracted. Check get_main_freq_traj, "
            "active bins, and duration thresholds."
        )

    cache_path = save_cache(records, output_dir)
    error_path = output_dir / "extraction_errors.txt"
    error_path.write_text("\n\n".join(errors), encoding="utf-8")

    summary = {
        "audio_files_scanned": len(audio_files),
        "contours_extracted": len(records),
        "files_with_errors": len(errors),
        "points_per_contour": args.points,
    }
    (output_dir / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Extracted {len(records):,} contours.")
    print(f"Saved: {cache_path}")
    print(f"Errors: {error_path} ({len(errors):,})")


def make_cluster_plot(
    cluster_id: int,
    member_indices: np.ndarray,
    contours_hz: np.ndarray,
    durations_s: np.ndarray,
    output_path: Path,
    max_examples: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed + cluster_id)
    selected = member_indices
    if len(selected) > max_examples:
        selected = rng.choice(selected, size=max_examples, replace=False)

    x = np.linspace(0.0, 1.0, contours_hz.shape[1])
    selected_khz = contours_hz[selected] / 1_000.0

    fig, ax = plt.subplots(figsize=(10, 7))
    for contour in selected_khz:
        ax.plot(x, contour, alpha=0.18, linewidth=0.8)

    ax.plot(x, np.mean(selected_khz, axis=0), linewidth=3, label="Mean")
    ax.plot(
        x,
        np.median(selected_khz, axis=0),
        linewidth=2,
        linestyle="--",
        label="Median",
    )
    ax.set_title(
        f"Cluster {cluster_id} | {len(member_indices):,} calls | "
        f"median duration {np.median(durations_s[member_indices]) * 1000:.1f} ms"
    )
    ax.set_xlabel("Normalized call time")
    ax.set_ylabel("Frequency (kHz)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def command_cluster(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache = load_cache(output_dir)

    features = cache["features"].astype(np.float64)
    contours_hz = cache["resampled_frequency_hz"].astype(np.float64)
    durations_s = cache["duration_s"].astype(np.float64)

    if len(features) < args.clusters:
        raise SystemExit(
            f"Only {len(features)} contours are available, fewer than "
            f"{args.clusters} clusters."
        )

    print(f"Clustering {len(features):,} contours into {args.clusters} groups...")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    n_components = max(
        2,
        min(args.pca_components, scaled.shape[1], len(scaled) - 1),
    )
    pca = PCA(
        n_components=n_components,
        whiten=True,
        random_state=args.seed,
    )
    reduced = pca.fit_transform(scaled)

    model = MiniBatchKMeans(
        n_clusters=args.clusters,
        init="k-means++",
        n_init=10,
        batch_size=min(args.batch_size, max(256, len(features))),
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    labels = model.fit_predict(reduced)
    distances = model.transform(reduced)
    distance_to_center = distances[np.arange(len(labels)), labels]

    assignments = pd.DataFrame(
        {
            "source_audio": cache["source_audio"].astype(str),
            "call_index": cache["call_index"].astype(int),
            "start_time_s": cache["start_time_s"].astype(float),
            "end_time_s": cache["end_time_s"].astype(float),
            "duration_s": durations_s,
            "cluster": labels.astype(int),
            "distance_to_cluster_center": distance_to_center,
        }
    )
    assignments_path = output_dir / ASSIGNMENTS_FILENAME
    assignments.to_csv(assignments_path, index=False)

    plots_dir = output_dir / "cluster_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for cluster_id in range(args.clusters):
        members = np.flatnonzero(labels == cluster_id)
        if not len(members):
            continue

        make_cluster_plot(
            cluster_id,
            members,
            contours_hz,
            durations_s,
            plots_dir / f"cluster_{cluster_id:02d}.png",
            args.plot_examples,
            args.seed,
        )

        summary_rows.append(
            {
                "cluster": cluster_id,
                "call_count": len(members),
                "median_duration_ms": np.median(durations_s[members]) * 1000,
                "median_start_frequency_khz": np.median(contours_hz[members, 0]) / 1000,
                "median_end_frequency_khz": np.median(contours_hz[members, -1]) / 1000,
                "median_range_khz": np.median(np.ptp(contours_hz[members], axis=1)) / 1000,
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "cluster_summary.csv",
        index=False,
    )

    label_path = output_dir / LABEL_MAP_FILENAME
    label_map = {
        str(cluster_id): f"cluster_{cluster_id}"
        for cluster_id in sorted(set(labels.tolist()))
    }
    label_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    sample_size = min(args.silhouette_sample, len(reduced))
    silhouette = None
    if sample_size >= 2 * args.clusters:
        rng = np.random.default_rng(args.seed)
        sample_indices = rng.choice(len(reduced), size=sample_size, replace=False)
        try:
            silhouette = float(
                silhouette_score(reduced[sample_indices], labels[sample_indices])
            )
        except ValueError:
            pass

    report = {
        "n_contours": len(features),
        "n_clusters": args.clusters,
        "pca_components": n_components,
        "pca_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
        "inertia": float(model.inertia_),
        "sample_silhouette_score": silhouette,
    }
    (output_dir / "clustering_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Assignments: {assignments_path}")
    print(f"Plots: {plots_dir}")
    print(f"Edit labels: {label_path}")


def sanitize_label(label: str) -> str:
    label = label.strip().lower().replace(" ", "_")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    cleaned = "".join(ch for ch in label if ch in allowed)
    return cleaned or "unlabeled"


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def command_label(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    assignments_path = output_dir / ASSIGNMENTS_FILENAME
    label_path = output_dir / LABEL_MAP_FILENAME

    if not assignments_path.exists() or not label_path.exists():
        raise SystemExit("Run cluster first, then edit cluster_labels.json.")

    assignments = pd.read_csv(assignments_path)
    raw_map = json.loads(label_path.read_text(encoding="utf-8"))
    label_map = {int(k): sanitize_label(v) for k, v in raw_map.items()}

    assignments["syllable_label"] = assignments["cluster"].map(label_map)
    if assignments["syllable_label"].isna().any():
        missing = sorted(
            assignments.loc[
                assignments["syllable_label"].isna(), "cluster"
            ].unique()
        )
        raise SystemExit(f"Missing labels for clusters: {missing}")

    cache = load_cache(output_dir)
    contours_dir = output_dir / "labeled_contours"
    contours_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = []
    for row_index, row in tqdm(
        assignments.iterrows(),
        total=len(assignments),
        desc="Writing contours",
    ):
        source = Path(row["source_audio"])
        label = str(row["syllable_label"])
        call_index = int(row["call_index"])

        label_dir = contours_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        output_path = unique_destination(
            label_dir / f"{label}_{source.stem}_call_{call_index:02d}.csv"
        )

        pd.DataFrame(
            {
                "time_s": np.asarray(cache["raw_time_s"][row_index], dtype=float),
                "frequency_hz": np.asarray(
                    cache["raw_frequency_hz"][row_index], dtype=float
                ),
                "source_audio": source.name,
                "call_index": call_index,
                "cluster": int(row["cluster"]),
                "syllable_label": label,
            }
        ).to_csv(output_path, index=False)
        generated_paths.append(str(output_path))

    assignments["labeled_contour_file"] = generated_paths

    if args.copy_audio:
        audio_dir = output_dir / "labeled_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        destinations = {}

        # If a recording has two calls, label the original audio using the
        # longest detected call.
        for source_audio, group in tqdm(
            assignments.groupby("source_audio"),
            desc="Copying audio",
        ):
            source = Path(source_audio)
            longest = group.loc[group["duration_s"].idxmax()]
            label = str(longest["syllable_label"])
            label_dir = audio_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            destination = unique_destination(
                label_dir / f"{label}_{source.name}"
            )
            shutil.copy2(source, destination)
            destinations[source_audio] = str(destination)

        assignments["labeled_audio_file"] = assignments["source_audio"].map(
            destinations
        )

    final_path = output_dir / "labeled_assignments.csv"
    assignments.to_csv(final_path, index=False)

    print(f"Labeled contours: {contours_dir}")
    if args.copy_audio:
        print(f"Labeled audio: {output_dir / 'labeled_audio'}")
        print("For two-call recordings, the longest call determines the audio label.")
    print(f"Final assignments: {final_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract, cluster, inspect, and label mouse USV contours."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_extract = subparsers.add_parser("extract")
    p_extract.add_argument("--audio-dir", required=True)
    p_extract.add_argument("--output-dir", default="usv_clusters")
    p_extract.add_argument("--points", type=int, default=100)
    p_extract.add_argument("--max-gap-bins", type=int, default=2)
    p_extract.add_argument("--min-active-bins", type=int, default=5)
    p_extract.add_argument("--min-duration", type=float, default=0.003)
    p_extract.add_argument("--max-duration", type=float, default=1.0)
    p_extract.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    p_extract.add_argument("--limit", type=int, default=None)
    p_extract.set_defaults(func=command_extract)

    p_cluster = subparsers.add_parser("cluster")
    p_cluster.add_argument("--output-dir", default="usv_clusters")
    p_cluster.add_argument("--clusters", type=int, default=15)
    p_cluster.add_argument("--pca-components", type=int, default=20)
    p_cluster.add_argument("--batch-size", type=int, default=2048)
    p_cluster.add_argument("--max-iter", type=int, default=300)
    p_cluster.add_argument("--plot-examples", type=int, default=100)
    p_cluster.add_argument("--silhouette-sample", type=int, default=5000)
    p_cluster.add_argument("--seed", type=int, default=42)
    p_cluster.set_defaults(func=command_cluster)

    p_label = subparsers.add_parser("label")
    p_label.add_argument("--output-dir", default="usv_clusters")
    p_label.add_argument("--copy-audio", action="store_true")
    p_label.set_defaults(func=command_label)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()