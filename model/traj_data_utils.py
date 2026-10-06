# loading + prepping the trajectory data for the lstm
# lives in model/ next to controller.py (moved out of model.ipynb so i'm not copy pasting the same cells everywhere lol)

import numpy as np
import torch
from scipy.interpolate import interp1d

# every trajectory gets stretched/squished to this many timesteps so they all line up
N_POINTS = 100

def resample_trajectory(freqs, n_points=N_POINTS):
    """turns a freq trajectory of any length into exactly n_points. returns None if it's too short to do that"""
    # gotta have at least 2 points to interpolate between, so skip anything shorter
    if len(freqs) < 2:
        return None

    # old x axis = where the real points are, new x axis = where we want the 100 points
    # both go from 0 to 1, so a long call and a short call end up the same length
    x_old = np.linspace(0, 1, len(freqs))
    x_new = np.linspace(0, 1, n_points)

    # heads up: this only looks at the freq values, never time_s. so if the tracker dropped frames
    # in the middle of a call, the chunks on either side get glued together and the gap just
    # disappears (might be making fake jumps, need to check this)
    return interp1d(x_old, freqs, kind="linear")(x_new)


def load_resampled(contours, ridge="bottom_or_main", n_points=N_POINTS):
    """resamples every trajectory in a contours dict (the one loaded from the pkl files).
    returns (trajectories, keys) in the same order, so trajectories[i] is the file keys[i].
    ridge can be "bottom_or_main" or "top" (top only exists for some files)"""
    trajectories, keys = [], []

    for key, data in contours.items():
        # each file looks like {"has_multiple_ridges", "bottom_or_main", "top"}
        # "top" is None when the file only has one ridge, so .get() and skip those
        ridge_data = data.get(ridge)
        if ridge_data is None:
            continue

        result = resample_trajectory(ridge_data["frequency_hz"], n_points)

        # only add the key if the trajectory actually made it through, that's what keeps
        # the two lists lined up (the keys list drifting out of sync is a pain to debug)
        if result is not None:
            trajectories.append(result)
            keys.append(key)

    return trajectories, keys


def prepare_batch(trajectories, pca, scaler, pca_dim=15):
    """makes the model input + targets for a batch of resampled trajectories.
    input = scaled pca vectors, targets = the real trajectories in hz"""
    X = np.array(trajectories)  # shape (batch, 100)

    # project onto the pca components, keep the first pca_dim of them, then scale
    # no pitch centering here (we decided against it) so pitch stays in the vectors
    vectors = scaler.transform(pca.transform(X)[:, :pca_dim])

    # float32 bc that's what the torch layers want
    return (
        torch.tensor(vectors, dtype=torch.float32),
        torch.tensor(X, dtype=torch.float32),
    )