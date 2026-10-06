# plotting stuff for the model notebook
# (this replaces model/utils.py, the two plot_activations functions are here now)
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch

from traj_data_utils import prepare_batch
from training_utils import predict_f0  # this import also puts the project root on sys.path


def plot_loss_curves(train_losses, test_losses, title="train vs. test loss"):
    """train vs test rmse per epoch, just pass in whatever train() returned"""
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.xlabel("epoch")
    plt.ylabel("RMSE (kHz)")
    plt.legend()
    plt.title(title)
    plt.show()


def plot_spectrogram_with_mft(idx, ax, keys, contours, trajectories, model, pca, scaler, pca_dim=15, 
                              n_fft=2048, hop_length=128):
    """spectrogram for file idx with the extracted ridge (cyan) and the model's prediction (green) on top.
    keys, contours and trajectories all need to come from the same load_resampled call
    (contours = the dict from the pkl file, trajectories = the resampled list)"""
    # imported in here so this file still loads even if data_prep isn't on the path yet
    # (netra's file, we only read from it, don't edit it)
    from data_prep.utils import find_audio_file, get_spectrogram

    key = keys[idx]
    audio_path = find_audio_file(key + ".WAV")  # change the extension if yours isn't .WAV

    times, freqs, S = get_spectrogram(audio_path, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # alpha + vmin = 45 so the background isn't too harsh and the lines actually like pop
    ax.pcolormesh(times * 1000, freqs / 1000, S_db, shading="auto", cmap="magma",
                  vmin=-45, vmax=0, alpha=0.75)

    # the "real" contour = what the tracker pulled out of the audio (bottom/main ridge)
    bottom = contours[key]["bottom_or_main"]
    raw_times = bottom["time_s"]
    raw_freqs = bottom["frequency_hz"]
    ax.plot(raw_times * 1000, raw_freqs / 1000, color="cyan", linewidth=2, label="extracted (real)")

    # model's prediction, stretched across the same time span as the real contour
    pca_vector, _ = prepare_batch([trajectories[idx]], pca, scaler, pca_dim)
    with torch.no_grad():
        predicted = predict_f0(model, pca_vector)
    pred_vals = predicted[0].numpy()
    pred_times_ms = np.linspace(raw_times.min(), raw_times.max(), len(pred_vals)) * 1000
    ax.plot(pred_times_ms, pred_vals / 1000, color="lime", linewidth=2, linestyle="--", label="predicted")

    ax.set_ylim(20, 125)  # usv range in khz (actually i dont remember fact check this)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("frequency (kHz)")
    ax.set_title(f"trajectory {idx} ({key})")
    ax.legend(loc="upper right")


# the two functions below were moved over from model/utils.py (docstrings shortened + lowercased)

def plot_activations(activations, syllable_id=None, title=None):
    """plots the 4 muscle activation curves (resp, pcaia, ct, ta) the lstm made for one sequence.
    activations = tensor of shape (1, seq_len, 4) or (seq_len, 4), straight from the model.
    if there's a batch dim it only plots the first item.
    title overrides the auto title, syllable_id is just for the auto title"""
    # drop the batch dim if there is one, detach from the graph, and go to numpy
    if activations.dim() == 3:
        activations = activations[0]
    activations = activations.detach().cpu().numpy()

    labels = ["resp_activity", "PCAIA_activity", "CT_activity", "TA_activity"]

    plt.figure(figsize=(10, 5))
    for i, label in enumerate(labels):
        plt.plot(activations[:, i], label=label)

    plt.xlabel("Timestep (% of syllable duration)")
    plt.ylabel("Activation (0-1)")
    plt.ylim(-0.05, 1.05)

    if title:
        plt.title(title)
    elif syllable_id is not None:
        plt.title(f"muscle activations: syllable class {syllable_id}")
    else:
        plt.title("muscle activations over time")

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_activations_multi(activations_dict):
    """same as plot_activations but side by side, one subplot per syllable class.
    handy for checking if different syllable ids give visibly different patterns.
    activations_dict = {syllable_id: activations tensor of shape (1, seq_len, 4) or (seq_len, 4)}"""
    labels = ["resp_activity", "PCAIA_activity", "CT_activity", "TA_activity"]
    n = len(activations_dict)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]  # subplots gives back a single ax (not a list) when n is 1

    for ax, (syllable_id, activations) in zip(axes, activations_dict.items()):
        if activations.dim() == 3:
            activations = activations[0]
        activations = activations.detach().cpu().numpy()

        for i, label in enumerate(labels):
            ax.plot(activations[:, i], label=label)

        ax.set_title(f"syllable {syllable_id}")
        ax.set_xlabel("Timestep")
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("activation (0-1)")
    axes[0].legend()
    plt.tight_layout()
    plt.show()