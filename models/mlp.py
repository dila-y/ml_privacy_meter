import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_shape, num_classes=10, dropout_p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_shape, 128)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, inputs):
        inputs = inputs.flatten(1)
        x = torch.tanh(self.fc1(inputs))
        x = self.dropout(x)   # our dropout defense is applied here
        outputs = self.fc2(x)
        return outputs

