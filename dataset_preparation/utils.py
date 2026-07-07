import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.ndimage import binary_closing, binary_opening
from pathlib import Path
from config import DATA_PATH

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

    return S_db, freqs, times


def track_ridge_tfridge_like(mag_usv, freqs_usv, active_bins, top_k=5, jump_penalty=1e-7):
    n_freq, n_time = mag_usv.shape
    candidates = []

    for t in range(n_time):
        if not active_bins[t]:
            candidates.append([])
            continue

        peaks, props = find_peaks(mag_usv[:, t])
        if len(peaks) == 0:
            peaks = np.array([np.argmax(mag_usv[:, t])])

        amps = mag_usv[peaks, t]
        best = peaks[np.argsort(amps)[-top_k:]]
        candidates.append(best)

    ridge = np.zeros(n_time)

    prev_freq = None

    for t in range(n_time):
        if len(candidates[t]) == 0:
            prev_freq = None
            continue

        best_score = -np.inf
        best_freq = 0

        for idx in candidates[t]:
            f = freqs_usv[idx]
            amp_score = np.log(mag_usv[idx, t] + 1e-12)

            if prev_freq is None:
                continuity_score = 0
            else:
                continuity_score = -jump_penalty * (f - prev_freq) ** 2

            score = amp_score + continuity_score

            if score > best_score:
                best_score = score
                best_freq = f

        ridge[t] = best_freq
        prev_freq = best_freq

    return ridge


def get_main_freq_traj(
    audio_path,
    freq_min=20000,
    freq_max=125000,
    n_fft=2048,
    hop_length=None,
    entropy_threshold=0.8,
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
    
    power = mag_usv ** 2

    # Normalize each time frame into a probability distribution.
    # Whistle-like frames concentrate power in a few bins -> low entropy.
    # Noise/silence spreads weak power around -> high entropy.
    prob = power / (np.sum(power, axis=0, keepdims=True) + 1e-12)

    entropy = -np.sum(prob * np.log2(prob + 1e-12), axis=0)

    # Normalize entropy to [0, 1]
    entropy = entropy / np.log2(prob.shape[0])

    # Smooth entropy over 3 frames
    if len(entropy) >= 3:
        entropy_smooth = np.convolve(entropy, np.ones(3) / 3, mode="same")
    else:
        entropy_smooth = entropy

    active_bins = entropy_smooth < entropy_threshold

    # Clean up isolated one-frame detections
    active_bins = binary_opening(active_bins, structure=np.ones(min_active_bins))

    # Fill tiny gaps inside a vocalization
    active_bins = binary_closing(active_bins, structure=np.ones(2))

    freq_traj = np.full(mag_usv.shape[1], silence_value, dtype=float)

    # Only run argmax where we believe there is real signal
    if np.any(active_bins):
        freq_traj = track_ridge_tfridge_like(
    mag_usv,
    freqs_usv,
    active_bins,
    top_k=5,
    jump_penalty=1e-7
)

    # Remove isolated impossible jumps
    for i in range(1, len(freq_traj) - 1):
        prev_f = freq_traj[i - 1]
        curr_f = freq_traj[i]
        next_f = freq_traj[i + 1]

        if prev_f == silence_value or curr_f == silence_value or next_f == silence_value:
            continue

        if (
            abs(curr_f - prev_f) > jump_threshold_hz
            and abs(curr_f - next_f) > jump_threshold_hz
        ):
            freq_traj[i] = (prev_f + next_f) / 2

    return times, freq_traj, active_bins


# ORIGINAL FUNCTION BELOW, KEPT FOR REFERENCE

# def get_main_freq_traj(audio_path, freq_min=20000, freq_max=125000,
#                          n_fft=1024, hop_length=None, energy_threshold=0.1):
#     """
#     Extracts the main (dominant) frequency trajectory from a USV audio file.

#     Parameters
#     ----------
#     audio_path : str or Path
#         Path to the .wav file.
#     freq_min : float
#         Minimum frequency (Hz) to consider — filters out non-USV noise. Default 20kHz.
#     freq_max : float
#         Maximum frequency (Hz) to consider. Default 125kHz (adjust based on your sample rate).
#     n_fft : int
#         FFT window size for the spectrogram.
#     hop_length : int or None
#         Number of samples between successive frames. Defaults to n_fft // 4 if None.
#     energy_threshold : float
#         Fraction of max energy below which a time bin is considered silence (frequency set to NaN).

#     Returns
#     -------
#     times : np.ndarray
#         Time values (seconds) for each point in the trajectory.
#     freq_traj : np.ndarray
#         Dominant frequency (Hz) at each time point. NaN where energy is below threshold.
#     """
#     times, freqs, magnitude = get_spectrogram(audio_path, n_fft=n_fft, hop_length=hop_length)

#     # Restrict to USV frequency range
#     freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
#     freqs_usv = freqs[freq_mask]
#     magnitude_usv = magnitude[freq_mask, :]

#     # Find dominant frequency at each time bin
#     max_energy_per_bin = magnitude_usv.max(axis=0)
#     dominant_freq_idx = np.argmax(magnitude_usv, axis=0)
#     freq_traj = freqs_usv[dominant_freq_idx]

#     # Mask out silent/low-energy bins as NaN
#     energy_cutoff = energy_threshold * max_energy_per_bin.max()
#     freq_traj = np.where(max_energy_per_bin >= energy_cutoff, freq_traj, np.nan)

#     return times, freq_traj
