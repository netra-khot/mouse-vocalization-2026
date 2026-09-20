import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.ndimage import binary_closing, binary_dilation, binary_opening
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import cv2

# finds the project root based where utils.py is located in the mouse vocal 2026 folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT)) # adds if it isnt already in sys.path

from config import DATA_PATH

TRAIN_PATH = Path(DATA_PATH) / "train"
TEST_PATH = Path(DATA_PATH) / "test"


def find_audio_file(filename): # check if the file exists

    for root in [TRAIN_PATH, TEST_PATH]:
        path = root / filename
        if path.exists():
            return path

    raise FileNotFoundError(f"Could not find {filename}") # same as throw in Java


def load_audio(wav_path, target_sr=None):
    audio, sr = sf.read(wav_path)

    if audio.ndim > 1: # need to check whether the file has multiple audio channels
        audio = audio.mean(axis=1)

    if target_sr is not None and sr != target_sr: # we only resample when a different target rate is put as param
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr # update sample rate of the file

    return audio.astype(np.float32), sr # return it as a float32 array


def bandpass_filter(audio, sr, low_hz=2500, high_hz=100000, order=3):
    nyq = sr / 2 # this is the highest frequency that can be seen in the audio file but high_hz can't be higher than this
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

    audio, sr = load_audio(audio_path, target_sr=target_sr) # load and resample

    if apply_bandpass:
        audio = bandpass_filter(audio, sr, low_hz=low_hz, high_hz=high_hz) # bandpass filter the audio to remove noise (only selected freq range)

    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window="hamming")) # split the audio into windows to check strength of the file
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft) # calc frequencies for each bin
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length) # calculate time for each frame

    return times, freqs, S


def quick_spectrogram(sig, sr, n_fft=1024, hop_length=128): # makes the spectrograms we see in audio_preprocess01
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


def load_spectrogram( # this entire method basically uses methods above to load and plot a spectrogram for given file
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

    n_times = magnitude.shape[1]
    freq_traj = np.full(n_times, np.nan, dtype=float)
    amplitude_traj = np.full(n_times, np.nan, dtype=float) # initializing getting amplitude traj like what Dr. Tripp was talking abt


    active_indices = np.flatnonzero(active_bins)

    if len(active_indices) == 0:
        return freq_traj, amplitude_traj

    split_locations = np.where(np.diff(active_indices) > 1)[0] + 1
    active_segments = np.split(active_indices, split_locations)

    for segment_times in active_segments:

        if len(segment_times) == 0:
            continue

        candidate_bins = []
        emission_scores = []

        segment_max_amplitude = np.max(
            magnitude[:, segment_times]
        )

        segment_max_db = 20 * np.log10(
            segment_max_amplitude + 1e-12
        )

        for t in segment_times:

            spectrum = magnitude[:, t]

            peaks, _ = find_peaks(spectrum)

            # if no local peak exists --> use strongest bin
            if len(peaks) == 0:
                peaks = np.array([np.argmax(spectrum)])

            peak_amplitudes = spectrum[peaks]

            strongest_order = np.argsort(peak_amplitudes)[::-1][:top_k]
            peaks = peaks[strongest_order]
            peak_amplitudes = peak_amplitudes[strongest_order]

            candidate_bins.append(peaks)

            candidate_db = (
                20 * np.log10(peak_amplitudes + 1e-12)
                - segment_max_db
            )

            candidate_score = np.clip(
                candidate_db,
                -60.0,
                0.0,
            )

            emission_scores.append(candidate_score)

        number_of_frames = len(segment_times)

        best_scores = [None] * number_of_frames
        backpointers = [None] * number_of_frames

        best_scores[0] = emission_scores[0].copy()
        backpointers[0] = np.full(
            len(candidate_bins[0]),
            -1,
            dtype=int,
        )

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

        for i, t in enumerate(segment_times):

            selected_index = selected_candidates[i]

            if selected_index < 0:
                continue

            frequency_bin = candidate_bins[i][selected_index]
            freq_traj[t] = freqs[frequency_bin] # storing frequency for each time point in mft
            amplitude_traj[t] = magnitude[frequency_bin, t] # storing amplitude for each time point in mft

    return freq_traj, amplitude_traj

def get_main_freq_traj(
    audio_path,
    freq_min=20000,
    freq_max=125000,
    n_fft=2048,
    hop_length=128,
    entropy_threshold=0.72,
    min_active_bins=2,
    silence_value=0.0,
):

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
        freq_traj = np.full_like(times, silence_value)
        amplitude_traj = np.zeros_like(times)
        active_bins = np.zeros_like(times, dtype=bool)
        return times, freq_traj, amplitude_traj, active_bins

    power = mag_usv ** 2
    prob = power / (np.sum(power, axis=0, keepdims=True) + 1e-12)

    entropy = -np.sum(prob * np.log2(prob + 1e-12), axis=0)

    # Normalize entropy to [0, 1]
    entropy = entropy / np.log2(prob.shape[0])

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

    #  Extend each detected vocalization by 2 frames on each side
    active_bins = binary_dilation(active_bins, structure=np.ones(5))

    freq_traj = np.full(mag_usv.shape[1], silence_value, dtype=float)
    amplitude_traj = np.full(mag_usv.shape[1], silence_value, dtype=float)

    # only run argmax where we think there is a real signal
    if np.any(active_bins):
        freq_traj, amplitude_traj = track_ridge_tfridge_like(
            magnitude=mag_usv,
            freqs=freqs_usv,
            active_bins=active_bins,
            top_k=8,
            jump_penalty=0.25,
            max_jump_hz=None,
        )


    return times, freq_traj, amplitude_traj, active_bins


def show_spectrogram_batch(
    file_df,
    batch_number=0,
    batch_size=20,
    random_state=42,
    freq_max_khz=125,
):

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

#CSV version
def export_mft(audio_path, output_dir):
    """
    Extract and save the MFT for a single WAV file.
    """

    times, freq_traj, amplitude_traj, active_bins = get_main_freq_traj(audio_path)

    df = pd.DataFrame({
        "time_s": times,
        "frequency_hz": freq_traj,
        "amplitude": amplitude_traj,
        "active": active_bins.astype(int),
    })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / (Path(audio_path).stem + ".csv")

    df.to_csv(output_file, index=False)

    return output_file

#Pickle version
def export_mft_pickle(audio_files, output_file):
    """
    Export MFT contours for multiple audio files into a single pickle file.
    """

    contours = {}

    for audio_path in audio_files:
        times, freq_traj, amplitude_traj, active_bins = get_main_freq_traj(audio_path)

        # Keep only the active portion of the contour
        contours[Path(audio_path).stem] = {
            "time_s": times[active_bins],
            "frequency_hz": freq_traj[active_bins],
            "amplitude": amplitude_traj[active_bins],
        }
        
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "wb") as f:
        pickle.dump(contours, f)

    return output_file


def check_mft_quality(
    audio_path,
    large_jump_hz=15_000,
    reversal_window=5,
    min_active_points=8,
):
    """
    Flag suspicious MFTs without treating every legitimate
    frequency jump as an error.
    """

    _, freq_traj, _, active_bins = get_main_freq_traj(audio_path)

    active_freq = np.asarray(freq_traj[active_bins], dtype=float)
    reasons = []

    if len(active_freq) == 0:
        return ["no_contour"]

    if np.any(~np.isfinite(active_freq)):
        reasons.append("contains_nan")

    valid_freq = active_freq[
        np.isfinite(active_freq) & (active_freq > 0)
    ]

    if len(valid_freq) < min_active_points:
        reasons.append("too_short")
        return sorted(set(reasons))

    frequency_changes = np.diff(valid_freq)

    jump_indices = np.flatnonzero(
        np.abs(frequency_changes) >= large_jump_hz
    )

    for i in range(len(jump_indices)):
        first_index = jump_indices[i]
        first_change = frequency_changes[first_index]

        for j in range(i + 1, len(jump_indices)):
            second_index = jump_indices[j]

            if second_index - first_index > reversal_window:
                break

            second_change = frequency_changes[second_index]

            # Large drop followed by large rise, or vice versa.
            if np.sign(first_change) != np.sign(second_change):
                reasons.append("jump_reversal")
                break

        if "jump_reversal" in reasons:
            break

    for start in range(len(frequency_changes)):
        end = min(
            start + reversal_window,
            len(frequency_changes),
        )

        jumps_in_window = np.sum(
            np.abs(frequency_changes[start:end]) >= large_jump_hz
        )

        if jumps_in_window >= 2:
            reasons.append("multiple_nearby_jumps")
            break

    return sorted(set(reasons))

def flag_mft_dataset(
    audio_files,
    output_csv="flagged_mfts.csv",
):

    rows = []

    for audio_path in audio_files:
        try:
            reasons = check_mft_quality(audio_path)

            if reasons:
                rows.append({
                    "filename": Path(audio_path).name,
                    "full_path": str(audio_path),
                    "reasons": "; ".join(reasons),
                })

        except Exception as exc:
            rows.append({
                "filename": Path(audio_path).name,
                "full_path": str(audio_path),
                "reasons": f"error: {type(exc).__name__}: {exc}",
            })

    flagged_df = pd.DataFrame(rows)
    flagged_df.to_csv(output_csv, index=False)

    print(f"Flagged {len(flagged_df)} files.")
    print(f"Saved to: {output_csv}")

    return flagged_df


def get_cv_freq_traj(path, threshold_db=-35):
    audio, sr = load_audio(path, target_sr=240000)
    audio = bandpass_filter(audio, sr)

    S_db, freqs, times = quick_spectrogram(audio, sr)
    S = librosa.db_to_amplitude(S_db)

    mask = (S_db > threshold_db).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    freq_traj = np.full(len(times), np.nan)
    amplitude_traj = np.full(len(times), np.nan)

    for t in range(len(times)):
        rows = np.flatnonzero(mask[:, t])
        if len(rows):
            bin_index = rows[np.argmax(S[rows, t])]
            freq_traj[t] = freqs[bin_index]
            amplitude_traj[t] = S[bin_index, t]

    return times, freq_traj, amplitude_traj