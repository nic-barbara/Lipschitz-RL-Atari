"""
Orthogonal layers from https://github.dev/acfr/LBDN/blob/main/layer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from liprl.networks.lbdn import cayley, StridedConv, fft_shift_matrix


class OrthogonLin(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super().__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.Q = None
            
    def forward(self, x):
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        y = F.linear(self.scale * x, Q, self.bias)
        return y


class OrthogonConvLin(StridedConv, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        args = list(args)
        if 'stride' in kwargs and kwargs['stride'] == 2:
            args = list(args)
            args[0] = 4 * args[0] # 4x in_channels
            if len(args) == 3:
                args[2] = max(1, args[2] // 2) 
                kwargs['padding'] = args[2] // 2 
            elif 'kernel_size' in kwargs:
                kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
                kwargs['padding'] = kwargs['kernel_size'] // 2
        scale = 1.0
        if 'scale' in kwargs:
            scale = kwargs['scale']
            del kwargs['scale']
        args = tuple(args)
        super().__init__(*args, **kwargs)
        self.scale = scale 
        self.register_parameter('alpha', None)
        self.Qfft = None
    
    def forward(self, x):
        x = self.scale * x 
        cout, cin, _, _ = self.weight.shape
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cin, batches)
        if self.training or self.Qfft is None or self.alpha is None:
            wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(cout, cin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
            if self.alpha is None:
                self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))
            self.Qfft = cayley(self.alpha * wfft / wfft.norm())
        Qfft = self.Qfft if self.training else self.Qfft.detach()
        yfft = (Qfft @ xfft).reshape(n, n // 2 + 1, cout, batches)
        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1))
        if self.bias is not None:
            y += self.bias[:, None, None]
        return y
