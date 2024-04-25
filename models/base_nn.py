import torch
import torch.nn as nn


class NN_model(torch.nn.Module):
    def __init__(self, input_dims, output_classes=250):
        super(NN_model, self).__init__()
        self.linear1 = nn.Linear(input_dims, 512)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(512, output_classes)
        
    def forward(self, x):
        x = self.linear1(x)
        x= self.activation1(x)
        x = self.linear2(x)

        return x