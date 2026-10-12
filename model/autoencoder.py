# sequence autoencoder for the usv trajectories (dr. tripp's suggestion, paper = devries + taylor 2017)

# idea: an encoder lstm reads the whole 100 point trajectory and squeezes it into a small latent vector,
# then a decoder lstm rebuilds the trajectory from that vector. it's basically a learned (nonlinear)
# version of pca, so hopefully it keeps the sharp jumps that pca smooths out

# lives in model/ next to controller.py. nothing in here touches the physics chain or the controller
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# lstms train badly on values like 80,000 so we divide hz by this to get roughly 0 to 1
# (the loss multiplies it back out so everything is still reported in khz)
HZ_SCALE = 100_000.0


class SeqAutoencoder(nn.Module):
    """encoder lstm -> latent vector -> decoder lstm. input and output are both (batch, 100) scaled trajectories"""

    def __init__(self, latent_dim=15, hidden_dim=64, num_layers=2, dropout=0.2,
                 seq_len=100, reverse_input=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.reverse_input = reverse_input

        # encoder: reads one freq value per timestep (input_size=1), 2 stacked lstm layers like the paper
        self.encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=dropout)

        # squeeze the encoder's final hidden state down to the latent vector (this is the "pca vector" now)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        # decoder: gets the latent vector as its input at EVERY timestep (the paper says this was
        # critical for good reconstructions). same trick as your controller's repeat, so no teacher forcing needed
        self.decoder = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=dropout)

        # one number out per timestep = the rebuilt (scaled) frequency
        self.out = nn.Linear(hidden_dim, 1)

    def encode(self, x):
        """x: (batch, seq_len) scaled trajectories -> (batch, latent_dim)"""
        # the paper reversed the input order and got faster training + better results
        # (the last thing the encoder reads is the first thing the decoder has to output)
        if self.reverse_input:
            x = x.flip(1)
        x = x.unsqueeze(-1)              # (batch, seq_len, 1) because lstm wants a feature dim
        _, (h, _) = self.encoder(x)      # h is (num_layers, batch, hidden_dim), one per layer
        return self.to_latent(h[-1])     # use the top layer's final hidden state

    def decode(self, z):
        """z: (batch, latent_dim) -> (batch, seq_len) scaled trajectories"""
        # copy the latent vector across all timesteps so the decoder sees it every step
        z_seq = z.unsqueeze(1).expand(-1, self.seq_len, -1)   # (batch, seq_len, latent_dim)
        out, _ = self.decoder(z_seq)                           # (batch, seq_len, hidden_dim)
        return self.out(out).squeeze(-1)                       # (batch, seq_len)

    def forward(self, x):
        return self.decode(self.encode(x))


def recon_loss_khz(recon, x):
    """rmse in khz. both inputs are scaled by HZ_SCALE, so multiply back out and divide by 1000.
    same math as rmse_loss_khz in training_utils (square, average everything, one sqrt) so numbers line up"""
    return torch.sqrt(F.mse_loss(recon, x)) * (HZ_SCALE / 1000)


def _to_tensor(trajectories):
    """list of (100,) hz arrays -> scaled float32 tensor of shape (n, 100)"""
    return torch.tensor(np.stack(trajectories) / HZ_SCALE, dtype=torch.float32)


def evaluate_ae(model, X, batch_size=512):
    """average batch rmse (khz) over a scaled tensor X, no gradients"""
    model.eval()  # turns dropout off
    losses = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            losses.append(recon_loss_khz(model(batch), batch).item())
    model.train()  # flip back so the next epoch trains normally
    return sum(losses) / len(losses)


def train_autoencoder(model, optimizer, train_trajectories, test_trajectories,
                      epochs=40, batch_size=128, scheduler=None):
    """trains the autoencoder to rebuild trajectories. returns (train_losses, test_losses) in khz, one per epoch.
    stop button ends it early but you still get the losses so far.
    unlike the controller loop this SHUFFLES the batches every epoch (standard for autoencoders)"""
    X_train = _to_tensor(train_trajectories)
    X_test = _to_tensor(test_trajectories)
    train_losses, test_losses = [], []

    try:
        for epoch in range(epochs):
            epoch_losses = []
            perm = torch.randperm(len(X_train))  # new random order each epoch

            # one bar per epoch, results printed underneath (same style as train())
            batch_bar = tqdm(range(0, len(X_train), batch_size),
                             desc=f"epoch {epoch + 1}/{epochs}", leave=True)
            for i in batch_bar:
                batch = X_train[perm[i:i + batch_size]]

                # usual routine: clear grads, forward, loss vs the input itself, backprop, update
                optimizer.zero_grad()
                loss = recon_loss_khz(model(batch), batch)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
            batch_bar.close()

            train_loss = sum(epoch_losses) / len(epoch_losses)
            test_loss = evaluate_ae(model, X_test)

            # same lr scheduler idea as the controller: cut lr when test loss stops improving
            if scheduler is not None:
                scheduler.step(test_loss)

            # don't delete these two (that's what bit us last time)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            tqdm.write(f"  train = {train_loss:.4f} kHz, test = {test_loss:.4f} kHz, "
                       f"lr = {optimizer.param_groups[0]['lr']:.2e}")
    except KeyboardInterrupt:
        print(f"stopped early after {len(train_losses)} epochs")

    return train_losses, test_losses


def reconstruct_all(model, trajectories, batch_size=512):
    """autoencoder reconstructions in hz, numpy array of shape (n, 100)"""
    X = _to_tensor(trajectories)
    model.eval()
    recons = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            recons.append(model(X[i:i + batch_size]).numpy())
    model.train()
    return np.concatenate(recons) * HZ_SCALE  # back to hz


def _rmse_khz(a, b):
    """pooled rmse in khz over every point (same as rmse_khz in training_utils, copied here
    so this file doesn't need the physics code to import)"""
    return np.sqrt(np.mean((a - b) ** 2)) / 1000


def compare_with_pca(model, trajectories, pca, k, jump_percentile=90):
    """THE comparison: autoencoder vs pca reconstruction, overall and on the jumpiest trajectories.
    k = how many pca components to keep. set it equal to the autoencoder's latent_dim or it's not a fair fight"""
    X = np.stack(trajectories)

    # autoencoder: trajectory -> latent -> trajectory
    ae_recon = reconstruct_all(model, trajectories)

    # pca: same idea, zero out the components past k then inverse transform
    vectors = pca.transform(X)
    vectors[:, k:] = 0
    pca_recon = pca.inverse_transform(vectors)

    # same jump subset definition as diagnose(): biggest frame to frame change, top 10%
    max_jump = np.max(np.abs(np.diff(X, axis=1)), axis=1)
    jump_mask = max_jump >= np.percentile(max_jump, jump_percentile)

    print(f"PCA ({k} comps) vs real (all): {_rmse_khz(pca_recon, X):.3f} kHz RMSE")
    print(f"autoencoder ({model.latent_dim} dims) vs real (all): {_rmse_khz(ae_recon, X):.3f} kHz RMSE")
    print(f"PCA ({k} comps) vs real (jump subset): {_rmse_khz(pca_recon[jump_mask], X[jump_mask]):.3f} kHz RMSE")
    print(f"autoencoder ({model.latent_dim} dims) vs real (jump subset): {_rmse_khz(ae_recon[jump_mask], X[jump_mask]):.3f} kHz RMSE")

    return {"ae_recon": ae_recon, "pca_recon": pca_recon, "jump_mask": jump_mask}


class AutoencoderEncoder:
    """ONLY for step 4 (if the autoencoder wins). wraps a trained autoencoder so it looks like a sklearn pca
    (transform / inverse_transform), so prepare_batch, train, evaluate and diagnose work unchanged
    when you pass this in place of `pca`. the encoder is frozen (everything runs under no_grad)"""

    def __init__(self, model, batch_size=512):
        self.model = model
        self.batch_size = batch_size
        self.n_components_ = model.latent_dim

    def transform(self, X):
        """(n, 100) trajectories in hz -> (n, latent_dim) latent vectors"""
        X = np.asarray(X, dtype=np.float32) / HZ_SCALE
        was_training = self.model.training
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                outs.append(self.model.encode(torch.from_numpy(X[i:i + self.batch_size])).numpy())
        if was_training:
            self.model.train()
        return np.concatenate(outs)

    def inverse_transform(self, Z):
        """(n, latent_dim) latent vectors -> (n, 100) trajectories in hz"""
        Z = np.asarray(Z, dtype=np.float32)
        was_training = self.model.training
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(Z), self.batch_size):
                outs.append(self.model.decode(torch.from_numpy(Z[i:i + self.batch_size])).numpy())
        if was_training:
            self.model.train()
        return np.concatenate(outs) * HZ_SCALE

import torch.nn as nn

class MLPAutoencoder(nn.Module):
    """plain feed forward baseline. trajectories are always 100 points so no lstm needed"""
    def __init__(self, latent_dim=15, hidden_dim=128, seq_len=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, seq_len),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))