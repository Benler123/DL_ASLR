import torch
import torch.nn as nn

class LSTM_model(nn.Module):
    def __init__(self, num_landmarks, hidden_size, num_layers, output_classes=250):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * num_landmarks, output_classes)

    def forward(self, x):
        batch_size, seq_len, num_landmarks, coord_dim = x.size()
        x = x.reshape(batch_size * seq_len, num_landmarks, coord_dim)

        h0 = torch.zeros(self.num_layers, batch_size * seq_len, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size * seq_len, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(batch_size, seq_len, num_landmarks * self.hidden_size)
        out = out[:, -1, :]  # Get the last hidden state
        out = self.fc(out)

        return out