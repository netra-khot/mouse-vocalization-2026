import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.ndimage import binary_closing, binary_opening
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PATH

print(plt)

TRAIN_PATH = Path(DATA_PATH) / "train"
TEST_PATH = Path(DATA_PATH) / "test"


def find_audio_file(filename):

    for root in [TRAIN_PATH, TEST_PATH]:
        path = root / filename
        if path.exists():
            return path

    raise FileNotFoundError(f"Could not find {filename}")


def load_audio(wav_path, target_sr=None):
    audio, sr = sf.read(wav_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if target_sr is not None and sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio.astype(np.float32), sr


def bandpass_filter(audio, sr, low_hz=2500, high_hz=100000, order=3):
    nyq = sr / 2
    high_hz = min(high_hz, nyq * 0.98)
    low_hz = min(low_hz, high_hz * 0.5)

    sos = butter(order, [low_hz, high_hz], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, audio)


def get_spectrogram(
    audio_path,
    n_fft=2048,
    hop_length=None,
    target_sr=240000,
    apply_bandpass=True,
    low_hz=2500,
    high_hz=100000,
):
    if hop_length is None:
        hop_length = n_fft  # 0% overlap

    audio, sr = load_audio(audio_path, target_sr=target_sr)

    if apply_bandpass:
        audio = bandpass_filter(audio, sr, low_hz=low_hz, high_hz=high_hz)

    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window="hamming"))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)

    return times, freqs, S


def quick_spectrogram(sig, sr, n_fft=1024, hop_length=128):
    S = np.abs(
        librosa.stft(
            sig,
            n_fft=n_fft,
            hop_length=hop_length,
            window="hamming",
        )
    )

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(
        np.arange(S.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )

    return librosa.amplitude_to_db(S, ref=np.max), freqs, times

def load_spectrogram(
    filename,
    filtered=False,
    target_sr=None,
    low_hz=2500,
    high_hz=100000,
    n_fft=1024,
    hop_length=128,
    plot=False,
    figsize=(16, 4),
):
    path = find_audio_file(filename)

    audio, sr = load_audio(path, target_sr=target_sr)

    if filtered:
        audio = bandpass_filter(audio, sr, low_hz, high_hz)

    S_db, freqs, times = quick_spectrogram(
        audio,
        sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    if plot:
        plt.figure(figsize=figsize)
        plt.pcolormesh(
            times * 1000,
            freqs / 1000,
            S_db,
            shading="auto",
            cmap="magma",
            vmin=-60,
            vmax=0,
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Frequency (kHz)")
        plt.title(filename)
        plt.tight_layout()
        plt.show()

    return S_db, freqs, times

def track_ridge_tfridge_like(
    magnitude,
    freqs,
    active_bins,
    top_k=8,
    jump_penalty=0.08,
    max_jump_hz=None,
):
    """
    Track the strongest globally consistent frequency ridge using
    dynamic programming.

    Parameters
    ----------
    magnitude : np.ndarray
        Spectrogram magnitude with shape:
        (number_of_frequencies, number_of_time_frames)

    freqs : np.ndarray
        Frequency value for each spectrogram row.

    active_bins : np.ndarray
        Boolean array indicating which time frames contain a detected
        vocalization.

    top_k : int
        Maximum number of peak candidates retained per active frame.

    jump_penalty : float
        Penalty applied per kHz of frequency movement between adjacent
        frames. Larger values favor smoother trajectories.

    max_jump_hz : float or None
        Optional maximum allowed frequency change between adjacent frames.
        Leave as None initially so genuine frequency jumps remain possible.

    Returns
    -------
    np.ndarray
        One frequency value per time frame. Inactive frames contain NaN.
    """

    n_freqs, n_times = magnitude.shape
    freq_traj = np.full(n_times, np.nan, dtype=float)

    active_indices = np.flatnonzero(active_bins)

    if len(active_indices) == 0:
        return freq_traj

    # ---------------------------------------------------------
    # Split active frames into separate contiguous segments.
    # This prevents the tracker from connecting vocalizations
    # across silence.
    # ---------------------------------------------------------
    split_locations = np.where(np.diff(active_indices) > 1)[0] + 1
    active_segments = np.split(active_indices, split_locations)

    for segment_times in active_segments:

        if len(segment_times) == 0:
            continue

        candidate_bins = []
        emission_scores = []

        # -----------------------------------------------------
        # Find candidate spectral peaks in every frame.
        # -----------------------------------------------------
        for t in segment_times:

            spectrum = magnitude[:, t]

            peaks, _ = find_peaks(spectrum)

            # If no local peak exists, use the strongest bin.
            if len(peaks) == 0:
                peaks = np.array([np.argmax(spectrum)])

            peak_amplitudes = spectrum[peaks]

            # Retain only the strongest top_k candidates.
            strongest_order = np.argsort(peak_amplitudes)[::-1][:top_k]
            peaks = peaks[strongest_order]
            peak_amplitudes = peak_amplitudes[strongest_order]

            candidate_bins.append(peaks)

            # Convert candidate amplitudes into relative dB scores.
            candidate_db = 20 * np.log10(peak_amplitudes + 1e-12)
            candidate_db -= np.max(candidate_db)

            # Map the relative values into approximately 0 to 1.
            candidate_score = np.clip(
                1.0 + candidate_db / 40.0,
                0.0,
                1.0,
            )

            emission_scores.append(candidate_score)

        number_of_frames = len(segment_times)

        # best_scores[i][j] is the best total score ending at
        # candidate j in segment frame i.
        best_scores = [None] * number_of_frames
        backpointers = [None] * number_of_frames

        # All candidates in the first frame are possible starting points.
        best_scores[0] = emission_scores[0].copy()
        backpointers[0] = np.full(
            len(candidate_bins[0]),
            -1,
            dtype=int,
        )

        # -----------------------------------------------------
        # Forward dynamic-programming pass.
        # -----------------------------------------------------
        for i in range(1, number_of_frames):

            current_candidates = candidate_bins[i]
            previous_candidates = candidate_bins[i - 1]

            current_freqs = freqs[current_candidates]
            previous_freqs = freqs[previous_candidates]

            current_scores = np.full(
                len(current_candidates),
                -np.inf,
                dtype=float,
            )

            current_backpointers = np.full(
                len(current_candidates),
                -1,
                dtype=int,
            )

            for current_index, current_frequency in enumerate(current_freqs):

                frequency_changes_hz = np.abs(
                    previous_freqs - current_frequency
                )

                transition_scores = (
                    best_scores[i - 1]
                    - jump_penalty * (frequency_changes_hz / 1000.0)
                )

                # Optionally prevent unrealistically large frame-to-frame
                # jumps while still allowing normal mode transitions.
                if max_jump_hz is not None:
                    transition_scores[
                        frequency_changes_hz > max_jump_hz
                    ] = -np.inf

                best_previous_index = np.argmax(transition_scores)
                best_previous_score = transition_scores[
                    best_previous_index
                ]

                if np.isfinite(best_previous_score):
                    current_scores[current_index] = (
                        best_previous_score
                        + emission_scores[i][current_index]
                    )

                    current_backpointers[current_index] = (
                        best_previous_index
                    )

            best_scores[i] = current_scores
            backpointers[i] = current_backpointers

        # -----------------------------------------------------
        # Backtrack from the best candidate in the final frame.
        # -----------------------------------------------------
        final_candidate = np.argmax(best_scores[-1])

        if not np.isfinite(best_scores[-1][final_candidate]):
            continue

        selected_candidates = np.full(
            number_of_frames,
            -1,
            dtype=int,
        )

        selected_candidates[-1] = final_candidate

        for i in range(number_of_frames - 1, 0, -1):

            selected_candidates[i - 1] = backpointers[i][
                selected_candidates[i]
            ]

            if selected_candidates[i - 1] < 0:
                break

        # -----------------------------------------------------
        # Convert selected candidate indices into frequencies.
        # -----------------------------------------------------
        for i, t in enumerate(segment_times):

            selected_index = selected_candidates[i]

            if selected_index < 0:
                continue

            frequency_bin = candidate_bins[i][selected_index]
            freq_traj[t] = freqs[frequency_bin]

    return freq_traj

# def track_ridge_tfridge_like(
#     magnitude,
#     freqs,
#     active_bins,
#     top_k=5,
#     jump_penalty=1e-7,
# ):
#     """
#     Track a frequency trajectory starting from the strongest active frame.

#     The trajectory is tracked:
#       1. Forward from the strongest frame
#       2. Backward from the strongest frame
#     """

#     n_freqs, n_times = magnitude.shape
#     freq_traj = np.full(n_times, np.nan)

#     # Find all active time frames.
#     active_indices = np.flatnonzero(active_bins)

#     if len(active_indices) == 0:
#         return freq_traj

#     # Find the strongest active time frame.
#     frame_energy = magnitude[:, active_indices].max(axis=0)
#     seed_position = np.argmax(frame_energy)
#     seed_time = active_indices[seed_position]

#     # Find peak candidates for every active frame.
#     candidates = []

#     for t in range(n_times):
#         if not active_bins[t]:
#             candidates.append(np.array([], dtype=int))
#             continue

#         peak_indices, _ = find_peaks(magnitude[:, t])

#         # If find_peaks finds nothing, use the strongest frequency bin.
#         if len(peak_indices) == 0:
#             peak_indices = np.array([np.argmax(magnitude[:, t])])

#         # Keep only the strongest top_k peaks.
#         peak_strengths = magnitude[peak_indices, t]
#         strongest_order = np.argsort(peak_strengths)[::-1][:top_k]
#         candidates.append(peak_indices[strongest_order])

#     # Initialize the trajectory at the strongest frame.
#     seed_candidates = candidates[seed_time]

#     if len(seed_candidates) == 0:
#         return freq_traj

#     seed_amplitudes = magnitude[seed_candidates, seed_time]
#     seed_index = seed_candidates[np.argmax(seed_amplitudes)]
#     freq_traj[seed_time] = freqs[seed_index]

#     # ---------------------------------------------------------
#     # Track forward from the strongest frame.
#     # ---------------------------------------------------------
#     previous_frequency = freq_traj[seed_time]

#     for t in range(seed_time + 1, n_times):
#         if not active_bins[t]:
#             continue

#         frame_candidates = candidates[t]

#         if len(frame_candidates) == 0:
#             continue

#         candidate_freqs = freqs[frame_candidates]
#         candidate_amplitudes = magnitude[frame_candidates, t]

#         scores = (
#             candidate_amplitudes
#             - jump_penalty * np.abs(candidate_freqs - previous_frequency)
#         )

#         best_candidate = frame_candidates[np.argmax(scores)]
#         freq_traj[t] = freqs[best_candidate]
#         previous_frequency = freq_traj[t]

#     # ---------------------------------------------------------
#     # Track backward from the strongest frame.
#     # ---------------------------------------------------------
#     previous_frequency = freq_traj[seed_time]

#     for t in range(seed_time - 1, -1, -1):
#         if not active_bins[t]:
#             continue

#         frame_candidates = candidates[t]

#         if len(frame_candidates) == 0:
#             continue

#         candidate_freqs = freqs[frame_candidates]
#         candidate_amplitudes = magnitude[frame_candidates, t]

#         scores = (
#             candidate_amplitudes
#             - jump_penalty * np.abs(candidate_freqs - previous_frequency)
#         )

#         best_candidate = frame_candidates[np.argmax(scores)]
#         freq_traj[t] = freqs[best_candidate]
#         previous_frequency = freq_traj[t]

#     return freq_traj

# def track_ridge_tfridge_like(mag_usv, freqs_usv, active_bins, top_k=5, jump_penalty=1e-9):
#     n_freq, n_time = mag_usv.shape
#     candidates = []

#     for t in range(n_time):
#         if not active_bins[t]:
#             candidates.append([])
#             continue

#         peaks, props = find_peaks(mag_usv[:, t])
#         if len(peaks) == 0:
#             peaks = np.array([np.argmax(mag_usv[:, t])])

#         amps = mag_usv[peaks, t]
#         best = peaks[np.argsort(amps)[-top_k:]]
#         candidates.append(best)

#     ridge = np.zeros(n_time)

#     prev_freq = None

#     for t in range(n_time):
#         if len(candidates[t]) == 0:
#             prev_freq = None
#             continue

#         best_score = -np.inf
#         best_freq = 0

#         for idx in candidates[t]:
#             f = freqs_usv[idx]
#             amp_score = np.log(mag_usv[idx, t] + 1e-12)

#             if prev_freq is None:
#                 continuity_score = 0
#             else:
#                 continuity_score = -jump_penalty * (f - prev_freq) ** 2

#             score = amp_score + continuity_score

#             if score > best_score:
#                 best_score = score
#                 best_freq = f

#         ridge[t] = best_freq
#         prev_freq = best_freq

#     return ridge


def get_main_freq_traj(
    audio_path,
    freq_min=20000,
    freq_max=125000,
    n_fft=2048,
    hop_length=128,
    entropy_threshold=0.76,
    min_active_bins=2,
    jump_threshold_hz=5000,
    silence_value=0.0,
):
    
    """
    Extract main frequency trajectory from a USV audio file.

    This avoids hallucinating frequencies during silence by:
    1. Computing a spectrogram.
    2. Detecting active USV frames using spectral entropy.
    3. Extracting the dominant frequency only in active frames.
    4. Setting silent frames to 0.
    5. Removing isolated jumps larger than jump_threshold_hz.

    Returns
    -------
    times : np.ndarray
        Time values in seconds.
    freq_traj : np.ndarray
        Main frequency trajectory in Hz. Silent frames are 0.
    active_bins : np.ndarray
        Boolean mask showing where a USV was detected.
    """

    if hop_length is None:
        hop_length = n_fft  # 0% overlap, matching Håkansson-style extraction

    times, freqs, magnitude = get_spectrogram(
        audio_path,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    freqs_usv = freqs[freq_mask]
    mag_usv = magnitude[freq_mask, :]

    if mag_usv.size == 0:
        return times, np.full_like(times, silence_value), np.zeros_like(times, dtype=bool)
    
    # frame_energy = mag_usv.max(axis=0)
    # seed = np.argmax(frame_energy * active_bins)

    power = mag_usv ** 2
    prob = power / (np.sum(power, axis=0, keepdims=True) + 1e-12)

    entropy = -np.sum(prob * np.log2(prob + 1e-12), axis=0)

    # Normalize entropy to [0, 1]
    entropy = entropy / np.log2(prob.shape[0])

    from scipy.ndimage import binary_opening, binary_closing, binary_dilation

    # Smooth entropy over 3 frames
    if len(entropy) >= 3:
        entropy_smooth = np.convolve(entropy, np.ones(3) / 3, mode="same")
    else:
        entropy_smooth = entropy

    # Initial entropy detection
    active_bins = entropy_smooth < entropy_threshold

    # Remove isolated detections
    active_bins = binary_opening(active_bins, structure=np.ones(min_active_bins))

    # Fill tiny gaps
    active_bins = binary_closing(active_bins, structure=np.ones(2))

    # NEW: extend each detected vocalization by 2 frames on each side
    active_bins = binary_dilation(active_bins, structure=np.ones(5))

    freq_traj = np.full(mag_usv.shape[1], silence_value, dtype=float)

    # Only run argmax where we thjnk there is real signal
    if np.any(active_bins):
#         freq_traj = track_ridge_tfridge_like(
#     mag_usv,
#     freqs_usv,
#     active_bins,
#     top_k=5,
#     jump_penalty=1e-9
# )
        freq_traj = track_ridge_tfridge_like(
    magnitude=mag_usv,
    freqs=freqs_usv,
    active_bins=active_bins,
    top_k=8,
    jump_penalty=0.02,
    max_jump_hz=None,
)

    # Remove isolated impossible jumps
    # for i in range(1, len(freq_traj) - 1):
    #     prev_f = freq_traj[i - 1]
    #     curr_f = freq_traj[i]
    #     next_f = freq_traj[i + 1]

    #     if prev_f == silence_value or curr_f == silence_value or next_f == silence_value:
    #         continue

    #     if (
    #         abs(curr_f - prev_f) > jump_threshold_hz
    #         and abs(curr_f - next_f) > jump_threshold_hz
    #     ):
    #         freq_traj[i] = (prev_f + next_f) / 2

    return times, freq_traj, active_bins


def show_spectrogram_batch(
    file_df,
    batch_number=0,
    batch_size=20,
    random_state=42,
    freq_max_khz=125,
):
    """
    Display one batch of spectrograms for visual inspection.

    batch_number=0 shows files 1-20
    batch_number=1 shows files 21-40
    etc.
    """

    shuffled_df = file_df.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    start = batch_number * batch_size
    end = min(start + batch_size, len(shuffled_df))
    batch_df = shuffled_df.iloc[start:end]

    if batch_df.empty:
        print("No more files to display.")
        return

    fig, axes = plt.subplots(5, 4, figsize=(18, 16))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, batch_df.iterrows()):
        audio, sr = load_audio(row["full_path"])

        S_db, freqs, times = quick_spectrogram(
            audio,
            sr,
            n_fft=1024,
            hop_length=128,
        )

        ax.pcolormesh(
            times * 1000,
            freqs / 1000,
            S_db,
            shading="auto",
            cmap="magma",
            vmin=-60,
            vmax=0,
        )

        ax.set_ylim(20, freq_max_khz)
        ax.set_title(row["filename"], fontsize=9)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (kHz)")

    # Hide unused panels in the final batch
    for ax in axes[len(batch_df):]:
        ax.axis("off")

    plt.suptitle(
        f"Spectrograms {start + 1}–{end} of {len(shuffled_df)}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()

    return batch_df