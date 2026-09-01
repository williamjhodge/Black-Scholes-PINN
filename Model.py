import torch
import torch.nn as nn
import numpy as np

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),      # input: (S, t)
            nn.Linear(64, 64), nn.Tanh(),     # use tanh activation function instead of ReLU to better capture the smoothness of the solution
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)                   # output: V
        )

    def forward(self, S, t):
        inputs = torch.cat([S, t], dim=1)
        return self.net(inputs)
