import torch
import torch.nn as nn

class LSTM_model(nn.Module):
    def __init__(self, num_landmarks, hidden_size, num_layers, output_classes, weight_decay=0.01, dropout_prob=0.5):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.weight_decay = weight_decay
        self.dropout_prob = dropout_prob
        
        self.lstm1 = nn.LSTM(input_size=num_landmarks*2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_classes)
    
    def forward(self, x):
        print(x.shape)
        batch_size, sequence_length, num_landmarks, _ = x.size()
        x = x.view(batch_size, sequence_length, -1)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm1(x, (h0, c0))
        out = self.dropout(out)
        out, _ = self.lstm2(out, (h0, c0))
        out = self.fc(out[:, -1, :])
        
        # L2 regularization
        l2_reg = torch.tensor(0.).to(x.device)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        
        return out, l2_reg * self.weight_decay
