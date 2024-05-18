"""
AoL layers from https://github.dev/acfr/LBDN/blob/main/layer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from https://github.dev/acfr/LBDN/blob/main/layer.py
class AolLin(nn.Module):
    def __init__(self, cin, cout, epsilon=1e-6, scale=1.0):
        super().__init__()
        self.scale = scale
        self.weights = nn.Parameter(torch.empty(cout, cin))
        self.bias = nn.Parameter(torch.empty(cout))

        nn.init.xavier_normal_(self.weights)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  

        self.epsilon = epsilon
        
        # Set up intermediate variables to avoid repeated computation
        self.T = None

    def forward(self, x):
        if self.training or (self.T is None):
            self.T = 1/torch.sqrt(torch.abs(self.weights.T @ self.weights).sum(1))
        x = self.scale * self.T * x 
        return F.linear(x, self.weights, self.bias)


class AolConvLin(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, epsilon=1e-6, scale=1.0):
        super().__init__()

        self.scale = scale
        self.kernel = nn.Parameter(torch.empty(cout, cin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(cout))
        self.padding = (kernel_size - 1) // 2
        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound) 

        self.epsilon = epsilon
        
        # Set up intermediate variables to avoid repeated computation
        self.kkt = None
        self.T = None

    def forward(self, x):
        if self.training or (self.kkt is None) or (self.T is None):
            self.kkt = F.conv2d(self.kernel, self.kernel, padding=self.kernel.shape[-1] - 1)
            self.T = 1 / torch.sqrt(torch.abs(self.kkt).sum((1, 2, 3)))
        res = F.conv2d(self.scale * x, self.kernel, padding=self.padding)
        res = self.T[None, :, None, None] * res + self.bias[:, None, None]
        return res
