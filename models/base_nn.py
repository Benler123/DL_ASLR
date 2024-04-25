import torch
import torch.nn as nn


class NN_model(torch.nn.Module):
    def __init__(self, input_dims, output_classes=250):
        super(NN_model, self).__init__()
        self.linear1 = nn.Linear(input_dims, 1024)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(1024, 512)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(512, output_classes)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.dropout(x)
        x = self.linear3(x)

        return x