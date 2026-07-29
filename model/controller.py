import torch
import torch.nn as nn

class LSTMController(nn.Module): 
    """
    LSTM controller to map the syllable class ID to a sequence of muscle activations
    (resp, PCAIA, CT, TA), one set per timestep (???), over a fixed 100-teimstep sequence
    (matching the resampled ground-truth main-frequency trajectory format.)"""

    # unsure on the seq_len
    # could try changing hidden_dim to 32...
    def __init__(self, num_syllable_classes, embedding_dim=16, hidden_dim=64, seq_len=100):
        super().__init__()
        self.seq_len = seq_len

        # syllable class ID -> dense vector
        self.embedding = nn.Embedding(num_syllable_classes, embedding_dim)

        # LSTM sees the same syllable embedding @ ea/ timestep
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)

        # hidden state -> 4 muscle activations per timestep (resp, PCAIA, CT, TA)
        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, syllable_id):
        """
        Args:
            syllable_id: tensor of shape (batch_size,) containing syllable class IDs

        Returns:
            activations: tensor of shape (batch, seq_len, 4) each
            value in [0,1] ordered as (resp, PCAIA, CT, TA)
        """
        batch_size = syllable_id.size[0]

        embedded = self.embedding(syllable_id)  # (batch_size, embedding_dim)
        embedded_repeated = embedded.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch_size, seq_len, embedding_dim)

        lstm_out, _ = self.lstm(embedded_repeated)  # (batch_size, seq_len, hidden_dim)

        raw_output = self.output_layer(lstm_out)  # (batch_size, seq_len, 4)
        activations = torch.sigmoid(raw_output)  # constrain to [0,1]

        return activations