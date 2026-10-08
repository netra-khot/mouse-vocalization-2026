# the physics chain, loss, eval, training loop and diagnostics for the lstm controller
# everything that used to be copy pasted all over model.ipynb lives here now

import sys
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

# model/ is one level below the project root, and the root is where QMC_mouseUSV/ and data_prep/ live
# so this makes the imports below work no matter what the notebook does with sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # only add it if it isn't already there
    sys.path.append(str(PROJECT_ROOT))

# ported hakansson physics stuff
from QMC_mouseUSV.subglottal_pressure import subglottal_pressure
from QMC_mouseUSV.glottal_area import glottal_area
from QMC_mouseUSV.impingement_length import impingement_length
from QMC_mouseUSV.jet_speed import jet_speed_from_pressure
from QMC_mouseUSV.USVfreq import USVfreq

from traj_data_utils import prepare_batch

def predict_f0(model, pca_vectors):
    """the one forward pass: pca vectors -> muscle activations -> physics chain -> f0 in hz.
    training, eval, diagnostics and the plots all go thru this now instead of each having their own copy"""
    # lstm gives 4 curves per trajectory (all between 0 and 1 bc of the sigmoid at the end)
    activations = model(pca_vectors)
    resp, PCAIA, CT, TA = (activations[..., i] for i in range(4))

    # now push them thru the hakansson model (frozen, only the lstm learns)
    pressure = subglottal_pressure(resp)                # resp -> subglottal pressure
    g_area = glottal_area(PCAIA, TA)                    # pcaia + ta -> glottal area
    imp_length = impingement_length(CT, TA)             # ct + ta -> impingement length
    speed = jet_speed_from_pressure(g_area, pressure)   # area + pressure -> jet speed
    return USVfreq(speed, imp_length)                   # jet speed + impingement length -> f0


def rmse_loss_khz(predicted_f0, targets):
    """rmse in khz. divide by 1000 first so the number is in khz (that's what we report everywhere)"""
    return torch.sqrt(torch.nn.functional.mse_loss(predicted_f0 / 1000, targets / 1000))


def evaluate(trajectories, model, pca, scaler, batch_size=128, pca_dim=15):
    """average batch rmse (khz) over a set of trajectories, no gradients"""
    model.eval()  # eval mode, not training
    losses = []

    with torch.no_grad():  # not training here so no need to track gradients
        for i in range(0, len(trajectories), batch_size):
            pca_vectors, targets = prepare_batch(
                trajectories[i:i + batch_size], pca, scaler, pca_dim
            )
            losses.append(rmse_loss_khz(predict_f0(model, pca_vectors), targets).item())

    model.train()  # flip it back so the next epoch trains normally
    return sum(losses) / len(losses)

def train(model, optimizer, train_trajectories, test_trajectories, pca,
          scaler, epochs=55, batch_size=128, pca_dim=15, scheduler=None):
    """trains the controller. returns (train_losses, test_losses), one rmse in khz per epoch.
    hitting the stop button ends training early but you still get the losses so far.
    scheduler is optional (ReduceLROnPlateau), it gets stepped on the test loss every epoch.
    batches go in the same order every epoch (no shuffling), same as the old notebook loop"""
    train_losses, test_losses = [], []

    try:
        for epoch in range(epochs):
            epoch_losses = []

            # one bar per epoch, fills up as the batches run and stays on screen after
            batch_bar = tqdm(
                range(0, len(train_trajectories), batch_size),
                desc=f"epoch {epoch + 1}/{epochs}",
                leave=True,
            )

            for i in batch_bar:
                pca_vectors, targets = prepare_batch(
                    train_trajectories[i:i + batch_size], pca, scaler, pca_dim
                )

                # usual torch routine: clear old grads, forward, loss, backprop, update
                optimizer.zero_grad()
                loss = rmse_loss_khz(predict_f0(model, pca_vectors), targets)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            batch_bar.close()

            train_loss = sum(epoch_losses) / len(epoch_losses)
            test_loss = evaluate(test_trajectories, model, pca, scaler, batch_size, pca_dim)

            # lr scheduler watches the test loss and cuts the lr when it stops improving
            if scheduler is not None:
                scheduler.step(test_loss)

            # these two lines are the ones that went missing, don't delete them
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            # tqdm.write puts the results on the line right under the bar
            tqdm.write(
                f"  train = {train_loss:.4f} kHz, test = {test_loss:.4f} kHz"
            )
    except KeyboardInterrupt:
        # so stopping a run early doesn't throw away the loss curves
        print(f"stopped early after {len(train_losses)} epochs")

    return train_losses, test_losses

def predict_all(model, trajectories, pca, scaler, pca_dim=15, batch_size=256):
    """predicted f0 (hz) for every trajectory, as a numpy array of shape (n, 100)"""
    model.eval()
    preds = []

    with torch.no_grad():
        # batching so it's fast (the old version did one trajectory at a time, which was slow)
        for i in range(0, len(trajectories), batch_size):
            pca_vectors, _ = prepare_batch(
                trajectories[i:i + batch_size], pca, scaler, pca_dim
            )
            preds.append(predict_f0(model, pca_vectors).numpy())

    model.train()
    return np.concatenate(preds)


def per_example_rmse_khz(a, b):
    """rmse in khz for each row of a vs the matching row of b"""
    return np.sqrt(np.mean((a - b) ** 2, axis=1)) / 1000


def rmse_khz(a, b):
    """pooled rmse in khz over every point in a vs b. same math as the training loss, so the numbers line up"""
    return np.sqrt(np.mean((a - b) ** 2)) / 1000


def diagnose(model, trajectories, pca, scaler, pca_dim=15, jump_percentile=90):
    """compares the model against the pca reconstruction, overall and on the jumpiest trajectories.
    everything printed is rmse in khz, same metric as train/test loss.
    returns the per example arrays too (for finding worst cases / plotting)"""
    X = np.stack(trajectories)

    # reconstruction: pca vector -> straight back to a trajectory, no model involved
    vectors = pca.transform(X)
    vectors[:, pca_dim:] = 0
    recon = pca.inverse_transform(vectors)

    preds = predict_all(model, trajectories, pca, scaler, pca_dim)

    # jumpiest = biggest change between two neighboring timesteps
    max_jump = np.max(np.abs(np.diff(X, axis=1)), axis=1)
    jump_mask = max_jump >= np.percentile(max_jump, jump_percentile)

    print(f"PCA reconstruction vs real (all): {rmse_khz(recon, X):.3f} kHz RMSE")
    print(f"model vs real (all): {rmse_khz(preds, X):.3f} kHz RMSE")
    print(f"model vs PCA reconstruction (all): {rmse_khz(preds, recon):.3f} kHz RMSE")
    print(f"PCA reconstruction vs real (jump subset): {rmse_khz(recon[jump_mask], X[jump_mask]):.3f} kHz RMSE")
    print(f"model vs real (jump subset): {rmse_khz(preds[jump_mask], X[jump_mask]):.3f} kHz RMSE")
    print(f"model vs PCA reconstruction (jump subset): {rmse_khz(preds[jump_mask], recon[jump_mask]):.3f} kHz RMSE")

    return {
        "model_vs_real_per_example": per_example_rmse_khz(preds, X),
        "pca_only_per_example": per_example_rmse_khz(recon, X),
        "jump_mask": jump_mask,
        "preds": preds,
        "recon": recon,
    }