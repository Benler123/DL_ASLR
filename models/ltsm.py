import torch
import torch.nn as nn

class LSTM_model(nn.Module):
    def __init__(self, input_dims, hidden_size, num_layers, output_classes=250):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dims, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Get the last hidden state
        out = self.fc(out)

        return out