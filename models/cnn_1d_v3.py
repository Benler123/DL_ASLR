import torch
import torch.nn as nn

class CNN1D_model(nn.Module):
    def __init__(self, num_landmarks, num_frames, output_classes=250):
        super(CNN1D_model, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=num_landmarks*2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.15)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.15)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.15)

        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        final_dim = num_frames // 8  
        self.fc1 = nn.Linear(256 * final_dim, 1024)  
        self.dropout_fc1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, output_classes)
        self.dropout_fc2 = nn.Dropout(0.3)
        

    def forward(self, x):
        # x = x.reshape(x.size(0), x.size(1), -1)
        # x = x.permute(0, 2, 1)
    

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = self.dropout_fc2(x)
        
        return x
