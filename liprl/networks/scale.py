import torch.nn as nn

class ScaleConst(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        return self.scale * x
