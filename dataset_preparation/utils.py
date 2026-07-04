import librosa
import numpy as np

def get_spectrogram(audio_path, n_fft=1024, hop_length=None):
    """
    Computes the spectrogram of a USV audio file.

    Returns
    -------
    times : np.ndarray
        Time values (seconds) for each frame.
    freqs : np.ndarray
        Frequency values (Hz) for each bin.
    magnitude : np.ndarray
        2D array of magnitude values, shape (n_freq_bins, n_time_frames).
    """
    y, sr = librosa.load(audio_path, sr=None)

    if hop_length is None:
        hop_length = n_fft // 4

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop_length)

    return times, freqs, magnitude

def get_main_freq_traj(audio_path, freq_min=20000, freq_max=125000,
                         n_fft=1024, hop_length=None, energy_threshold=0.1):
    """
    Extracts the main (dominant) frequency trajectory from a USV audio file.

    Parameters
    ----------
    audio_path : str or Path
        Path to the .wav file.
    freq_min : float
        Minimum frequency (Hz) to consider — filters out non-USV noise. Default 20kHz.
    freq_max : float
        Maximum frequency (Hz) to consider. Default 125kHz (adjust based on your sample rate).
    n_fft : int
        FFT window size for the spectrogram.
    hop_length : int or None
        Number of samples between successive frames. Defaults to n_fft // 4 if None.
    energy_threshold : float
        Fraction of max energy below which a time bin is considered silence (frequency set to NaN).

    Returns
    -------
    times : np.ndarray
        Time values (seconds) for each point in the trajectory.
    freq_traj : np.ndarray
        Dominant frequency (Hz) at each time point. NaN where energy is below threshold.
    """
    times, freqs, magnitude = get_spectrogram(audio_path, n_fft=n_fft, hop_length=hop_length)

    # Restrict to USV frequency range
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    freqs_usv = freqs[freq_mask]
    magnitude_usv = magnitude[freq_mask, :]

    # Find dominant frequency at each time bin
    max_energy_per_bin = magnitude_usv.max(axis=0)
    dominant_freq_idx = np.argmax(magnitude_usv, axis=0)
    freq_traj = freqs_usv[dominant_freq_idx]

    # Mask out silent/low-energy bins as NaN
    energy_cutoff = energy_threshold * max_energy_per_bin.max()
    freq_traj = np.where(max_energy_per_bin >= energy_cutoff, freq_traj, np.nan)

    return times, freq_traj