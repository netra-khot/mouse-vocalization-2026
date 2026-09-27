import torch
import torch.nn as nn

class LSTMController(nn.Module): 
    """
    LSTM controller to map the PCA vector to a sequence of muscle activations
    (resp, PCAIA, CT, TA), one set per timestep, over a fixed 100-timestep sequence
    (matching the resampled ground-truth main-frequency trajectory format.)"""

    def __init__(self, input_dim=11, embedding_dim=16, hidden_dim=128, seq_len=100):
        super().__init__()
        self.seq_len = seq_len

        # accepts a continuous pca vector (+ mean pitch) instead of a discrete syllable id
        self.embedding = nn.Linear(input_dim, embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, pca_vector):
        batch_size = pca_vector.size(0)

        embedded = self.embedding(pca_vector)
        embedded_repeated = embedded.unsqueeze(1).repeat(1, self.seq_len, 1)

        lstm_out, _ = self.lstm(embedded_repeated)

        raw_output = self.fc(lstm_out)
        activations = torch.sigmoid(raw_output)

        return activations