import torch
import torch.nn as nn

class LSTM_model(nn.Module):
    def __init__(self, num_landmarks, hidden_size, num_layers, output_classes):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=num_landmarks*2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_classes)
        
    def forward(self, x):
        # input = (batch_size, sequence_length, num_landmarks*2)
        batch_size, sequence_length, num_landmarks, _ = x.size()
        x = x.view(batch_size, sequence_length, -1)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        
        return out