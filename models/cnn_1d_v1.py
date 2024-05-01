import torch
import torch.nn as nn

class CNN1D_model(nn.Module):
    def __init__(self, num_landmarks, num_frames, output_classes=250):
        super(CNN1D_model, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=num_landmarks*2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        final_dim = num_frames // 2 // 2 // 2  
        self.fc1 = nn.Linear(256 * final_dim, 1024)
        self.fc2 = nn.Linear(1024, output_classes)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x

