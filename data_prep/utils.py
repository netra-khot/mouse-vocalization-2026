from os import times
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
    top_k=12,
    jump_penalty=0.03,
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
    n_fft=1024,
    hop_length=128,
    entropy_threshold=0.85,
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
            top_k=12,
            jump_penalty=0.03,
            max_jump_hz=None,
        )

    # remove isolated frequency spikes without deleting actual jumps
    # for i in range(1, len(freq_traj) - 1):
    #     if not np.all(np.isfinite(freq_traj[i - 1:i + 2])):
    #         continue

    #     jump_before = abs(freq_traj[i] - freq_traj[i - 1])
    #     jump_after = abs(freq_traj[i] - freq_traj[i + 1])
    #     neighbors_match = abs(freq_traj[i - 1] - freq_traj[i + 1]) < 5000

    #     if jump_before > 10000 and jump_after > 10000 and neighbors_match:
    #         freq_traj[i] = np.nan
    #         amplitude_traj[i] = np.nan
    #         active_bins[i] = False
    jump_indices = np.where(np.abs(np.diff(freq_traj)) > 10000)[0] + 1

    freq_traj[jump_indices] = np.nan
    amplitude_traj[jump_indices] = np.nan
    active_bins[jump_indices] = False

    return times, freq_traj, amplitude_traj, active_bins

def _track_ridge_branch(
    magnitude,
    freqs,
    active_bins,
    branch,
    bottom_freq_traj=None,
    top_k=12,
    jump_penalty=0.03,
    max_jump_hz=None,
    threshold_above_noise_db=12,
    noise_percentile=20,
    branch_preference_db=8,
    min_separation_hz=2000,
):
    n_times = magnitude.shape[1]

    freq_traj = np.full(n_times, np.nan)
    amplitude_traj = np.full(n_times, np.nan)

    magnitude_db = librosa.amplitude_to_db(
        magnitude,
        ref=np.max,
    )

    # estimate background at each frequency
    noise_floor_db = np.percentile(
        magnitude_db,
        noise_percentile,
        axis=1,
    )

    candidate_bins_by_time = []
    candidate_scores_by_time = []

    frequency_position = (
        (freqs - freqs.min())
        / (freqs.max() - freqs.min() + 1e-12)
    )

    for t in range(n_times):
        if not active_bins[t]:
            candidate_bins_by_time.append(np.array([], dtype=int))
            candidate_scores_by_time.append(np.array([]))
            continue

        peaks, _ = find_peaks(magnitude[:, t])

        if len(peaks) == 0:
            candidate_bins_by_time.append(np.array([], dtype=int))
            candidate_scores_by_time.append(np.array([]))
            continue

        db_above_noise = (
            magnitude_db[peaks, t]
            - noise_floor_db[peaks]
        )

        # Only retain peaks above background noise theshold
        keep = db_above_noise >= threshold_above_noise_db
        peaks = peaks[keep]
        db_above_noise = db_above_noise[keep]

        # Remove peaks much weaker than the strongest peak in this frame
        if len(peaks) > 0:
            strongest_peak_db = np.max(magnitude_db[peaks, t])

            maximum_drop_db = 20 if branch == "top" else 12

            strong_enough = (
                magnitude_db[peaks, t]
                >= strongest_peak_db - maximum_drop_db
            )

            peaks = peaks[strong_enough]
            db_above_noise = db_above_noise[strong_enough]

        # Choose the lowest strong peak for the bottom trajectory
        if branch == "bottom" and len(peaks) > 0:
            lowest_index = np.argmin(freqs[peaks])

            peaks = peaks[[lowest_index]]
            db_above_noise = db_above_noise[[lowest_index]]

        # so this is making sure the top ridge must be above the detected bottom ridge
        if (
            branch == "top"
            and bottom_freq_traj is not None
            and np.isfinite(bottom_freq_traj[t])
        ):
            keep = (
                freqs[peaks]
                >= bottom_freq_traj[t] + min_separation_hz
            )

            peaks = peaks[keep]
            db_above_noise = db_above_noise[keep]

        if len(peaks) == 0:
            candidate_bins_by_time.append(np.array([], dtype=int))
            candidate_scores_by_time.append(np.array([]))
            continue

        # Keep the strongest qualifying peaks
        order = np.argsort(db_above_noise)[::-1][:top_k]
        peaks = peaks[order]
        db_above_noise = db_above_noise[order]

        # Favor lower frequencies for bottom and higher ones for top
        if branch == "bottom":
            branch_score = (
                branch_preference_db
                * (1 - frequency_position[peaks])
            )
        else:
            branch_score = (
                branch_preference_db
                * frequency_position[peaks]
            )

        candidate_bins_by_time.append(peaks)
        candidate_scores_by_time.append(
            db_above_noise + branch_score
        )

    # Frames without qualifying peaks become gaps
    usable_bins = active_bins & np.array([
        len(peaks) > 0 for peaks in candidate_bins_by_time
    ])

    usable_indices = np.flatnonzero(usable_bins)

    if len(usable_indices) == 0:
        return freq_traj, amplitude_traj

    split_locations = (
        np.where(np.diff(usable_indices) > 1)[0] + 1
    )

    segments = np.split(usable_indices, split_locations)

    for segment_times in segments:
        number_of_frames = len(segment_times)

        if number_of_frames == 0:
            continue

        best_scores = [None] * number_of_frames
        backpointers = [None] * number_of_frames

        first_t = segment_times[0]

        best_scores[0] = candidate_scores_by_time[
            first_t
        ].copy()

        backpointers[0] = np.full(
            len(candidate_bins_by_time[first_t]),
            -1,
            dtype=int,
        )

        for i in range(1, number_of_frames):
            current_t = segment_times[i]
            previous_t = segment_times[i - 1]

            current_candidates = candidate_bins_by_time[current_t]
            previous_candidates = candidate_bins_by_time[previous_t]

            current_freqs = freqs[current_candidates]
            previous_freqs = freqs[previous_candidates]

            current_scores = np.full(
                len(current_candidates),
                -np.inf,
            )

            current_backpointers = np.full(
                len(current_candidates),
                -1,
                dtype=int,
            )

            for current_index, current_frequency in enumerate(
                current_freqs
            ):
                frequency_changes_hz = np.abs(
                    previous_freqs - current_frequency
                )

                transition_scores = (
                    best_scores[i - 1]
                    - jump_penalty
                    * (frequency_changes_hz / 1000)
                )

                if max_jump_hz is not None:
                    transition_scores[
                        frequency_changes_hz > max_jump_hz
                    ] = -np.inf

                best_previous_index = np.argmax(
                    transition_scores
                )

                best_previous_score = transition_scores[
                    best_previous_index
                ]

                if np.isfinite(best_previous_score):
                    current_scores[current_index] = (
                        best_previous_score
                        + candidate_scores_by_time[
                            current_t
                        ][current_index]
                    )

                    current_backpointers[current_index] = (
                        best_previous_index
                    )

            best_scores[i] = current_scores
            backpointers[i] = current_backpointers

        final_candidate = np.argmax(best_scores[-1])

        if not np.isfinite(
            best_scores[-1][final_candidate]
        ):
            continue

        selected = np.full(
            number_of_frames,
            -1,
            dtype=int,
        )

        selected[-1] = final_candidate

        for i in range(number_of_frames - 1, 0, -1):
            selected[i - 1] = backpointers[i][selected[i]]

            if selected[i - 1] < 0:
                break

        for i, t in enumerate(segment_times):
            if selected[i] < 0:
                continue

            frequency_bin = candidate_bins_by_time[t][
                selected[i]
            ]

            freq_traj[t] = freqs[frequency_bin]
            amplitude_traj[t] = magnitude[frequency_bin, t]

    return freq_traj, amplitude_traj

def track_bottom_ridge_tfridge_like(
    magnitude,
    freqs,
    active_bins,
    **kwargs,
):
    return _track_ridge_branch(
        magnitude=magnitude,
        freqs=freqs,
        active_bins=active_bins,
        branch="bottom",
        **kwargs,
    )

def track_top_ridge_tfridge_like(
    magnitude,
    freqs,
    active_bins,
    bottom_freq_traj,
    **kwargs,
):
    return _track_ridge_branch(
        magnitude=magnitude,
        freqs=freqs,
        active_bins=active_bins,
        branch="top",
        bottom_freq_traj=bottom_freq_traj,
        **kwargs,
    )

def get_dual_freq_traj(
    audio_path,
    freq_min=20000,
    freq_max=125000,
    n_fft=1024,
    hop_length=128,
    entropy_threshold=0.85,
    min_active_bins=2,
):
    times, freqs, magnitude = get_spectrogram(
        audio_path,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    freq_mask = (
        (freqs >= freq_min)
        & (freqs <= freq_max)
    )

    freqs_usv = freqs[freq_mask]
    mag_usv = magnitude[freq_mask, :]

    empty = np.full(len(times), np.nan)
    inactive = np.zeros(len(times), dtype=bool)

    if mag_usv.size == 0:
        return (
            times,
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
            inactive.copy(),
            inactive.copy(),
        )

    power = mag_usv ** 2

    probability = power / (
        np.sum(power, axis=0, keepdims=True) + 1e-12
    )

    entropy = -np.sum(
        probability * np.log2(probability + 1e-12),
        axis=0,
    )

    entropy /= np.log2(probability.shape[0])

    if len(entropy) >= 3:
        entropy = np.convolve(
            entropy,
            np.ones(3) / 3,
            mode="same",
        )

    detected_bins = entropy < entropy_threshold

    detected_bins = binary_opening(
        detected_bins,
        structure=np.ones(min_active_bins),
    )

    detected_bins = binary_closing(
        detected_bins,
        structure=np.ones(2),
    )

    detected_bins = binary_dilation(
        detected_bins,
        structure=np.ones(5),
    )

    # Convert the USV spectrogram to decibels
    magnitude_db = librosa.amplitude_to_db(
        mag_usv,
        ref=np.max,
    )

    # Estimate background noise at each frequency
    noise_floor_db = np.percentile(
        magnitude_db,
        20,
        axis=1,
    )

    multiple_peak_frames = 0
    current_run = 0
    longest_run = 0

    active_frame_count = np.sum(detected_bins)

    dual_frame_mask = np.zeros(len(times), dtype=bool)

    for t in range(len(times)):
        if not detected_bins[t]:
            current_run = 0
            continue

        peaks, _ = find_peaks(mag_usv[:, t])

        if len(peaks) < 2:
            current_run = 0
            continue

        db_above_noise = (
            magnitude_db[peaks, t]
            - noise_floor_db[peaks]
        )

        # Peaks must be at least 15 dB above background
        peaks = peaks[db_above_noise >= 15]

        if len(peaks) < 2:
            current_run = 0
            continue

        # Sort peaks from strongest to weakest
        order = np.argsort(
            magnitude_db[peaks, t]
        )[::-1]

        peaks = peaks[order]
        strongest_peak = peaks[0]

        second_ridge_found = False

        for second_peak in peaks[1:]:
            separation = abs(
                freqs_usv[second_peak]
                - freqs_usv[strongest_peak]
            )

            amplitude_difference = (
                magnitude_db[strongest_peak, t]
                - magnitude_db[second_peak, t]
            )

            # Require a strong second peak separated by 15 kHz
            if (
                separation >= 10000
                and amplitude_difference <= 12
            ):
                second_ridge_found = True
                break

        if second_ridge_found:
            dual_frame_mask[t] = True
            multiple_peak_frames += 1
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    dual_frame_mask = binary_opening(
        dual_frame_mask,
        structure=np.ones(3),
    )

    dual_frame_mask = binary_closing(
        dual_frame_mask,
        structure=np.ones(5),
    )

    minimum_multiple_frames = max(
        6,
        int(0.20 * active_frame_count),
    )

    has_multiple_ridges = (
        multiple_peak_frames >= minimum_multiple_frames
        and longest_run >= 6
    )

    print(
        "Multiple-ridge detection:",
        has_multiple_ridges,
        "| frames:",
        multiple_peak_frames,
        "| longest run:",
        longest_run,
    )

    # Use the original MFT method for ordinary single-ridge calls
    if not has_multiple_ridges:
        main_freq, main_amplitude = track_ridge_tfridge_like(
            magnitude=mag_usv,
            freqs=freqs_usv,
            active_bins=detected_bins,
            top_k=12,
            jump_penalty=0.03,
            max_jump_hz=None,
        )

        main_active = np.isfinite(main_freq)

        top_freq = np.full(len(times), np.nan)
        top_amplitude = np.full(len(times), np.nan)
        top_active = np.zeros(len(times), dtype=bool)

        return (
            times,
            main_freq,       # Stored in the bottom/main position
            main_amplitude,
            top_freq,
            top_amplitude,
            main_active,
            top_active,
        )

    bottom_freq, bottom_amplitude = (
        track_bottom_ridge_tfridge_like(
            magnitude=mag_usv,
            freqs=freqs_usv,
            active_bins=detected_bins & dual_frame_mask,
            threshold_above_noise_db=15,
            noise_percentile=20,
            branch_preference_db=25,
            top_k=12,
            jump_penalty=0.15,
        )
    )

    # replace sudden bottom-ridge jumps with gaps TODO could remove this
    bottom_jumps = np.where(
        np.abs(np.diff(bottom_freq)) > 10000
    )[0] + 1

    bottom_freq[bottom_jumps] = np.nan
    bottom_amplitude[bottom_jumps] = np.nan

    top_freq, top_amplitude = (
        track_top_ridge_tfridge_like(
            magnitude=mag_usv,
            freqs=freqs_usv,
            active_bins=detected_bins,
            bottom_freq_traj=bottom_freq,
            threshold_above_noise_db=10,
            noise_percentile=20,
            branch_preference_db=6,
            min_separation_hz=10000,
            top_k=12,
            jump_penalty=0.25,
        )
    ) 
    # Remove short downward branch switches from the top trajectory
    changes = np.diff(top_freq)

    drops = np.where(changes < -10000)[0] + 1
    rises = np.where(changes > 10000)[0] + 1

    for start in drops:
        possible_ends = rises[
            (rises > start) & (rises - start <= 35)
        ]

        if len(possible_ends):
            end = possible_ends[0]
            top_freq[start:end] = np.nan
            top_amplitude[start:end] = np.nan

    # Connect the surrounding upper-ridge sections
    top_freq = (
        pd.Series(top_freq)
        .interpolate(limit=35, limit_area="inside")
        .to_numpy()
    )
       

    # Remove short branch switches
    cleaned = []

    for freq, amplitude in [
        (bottom_freq, bottom_amplitude),
    ]:
        freq = freq.copy()
        amplitude = amplitude.copy()
        i = 1

        while i < len(freq) - 1:
            if not (
                np.isfinite(freq[i - 1])
                and np.isfinite(freq[i])
            ):
                i += 1
                continue

            baseline = freq[i - 1]

            if abs(freq[i] - baseline) <= 15000:
                i += 1
                continue

            start = i
            j = i + 1

            while (
                j < len(freq)
                and j - start <= 4
                and np.isfinite(freq[j])
                and abs(freq[j] - baseline) > 5000
            ):
                j += 1

            if (
                j < len(freq)
                and j - start <= 4
                and np.isfinite(freq[j])
                and abs(freq[j] - baseline) <= 5000
            ):
                freq[start:j] = np.nan
                amplitude[start:j] = np.nan

            i = max(j, i + 1)

        cleaned.append((freq, amplitude))

    bottom_freq, bottom_amplitude = cleaned[0]

    bottom_active = np.isfinite(bottom_freq)
    top_active = np.isfinite(top_freq)

    return (
        times,
        bottom_freq,
        bottom_amplitude,
        top_freq,
        top_amplitude,
        bottom_active,
        top_active,
    )

#CSV version
def export_mft_csv(audio_files, output_file):
    rows = []

    for audio_path in audio_files:
        features = extract_usv_features(audio_path)

        if features is not None:
            rows.append(features)

    df = pd.DataFrame(rows)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    return output_file

#Pickle version
def export_dual_mft_pickle(audio_files, output_file):
    contours = {}
    for audio_path in audio_files:
        (
            times,
            bottom_freq,
            bottom_amplitude,
            top_freq,
            top_amplitude,
            bottom_active,
            top_active,
        ) = get_dual_freq_traj(audio_path)

        if np.sum(bottom_active) < 2:
            continue

        bottom_times = times[bottom_active]
        bottom_data = {
            "time_s": bottom_times,
            "frequency_hz": bottom_freq[bottom_active],
            "amplitude": bottom_amplitude[bottom_active],
            "duration_s": bottom_times[-1] - bottom_times[0],
        }

        has_top = np.sum(top_active) >= 2
        top_data = None
        if has_top:
            top_times = times[top_active]

            top_data = {
                "time_s": top_times,
                "frequency_hz": top_freq[top_active],
                "amplitude": top_amplitude[top_active],
                "duration_s": top_times[-1] - top_times[0],
            }

        contours[Path(audio_path).stem] = {
            "has_multiple_ridges": has_top,
            "bottom_or_main": bottom_data,
            "top": top_data,
        }
    Path(output_file).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(output_file, "wb") as f:
        pickle.dump(contours, f)

    return output_file

