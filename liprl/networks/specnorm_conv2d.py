import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralNormalizationConv2D(nn.Module):
    """Implements spectral normalization for Conv2D layer based on [3].
    
    All code translated from TensorFlow implementation in:
    https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/normalization.py
  
    [3] Henry Gouk, Eibe Frank, Bernhard Pfahringer, Michael Cree.
    Regularisation of neural networks by enforcing lipschitz continuity.
    _arXiv preprint arXiv:1804.04368_, 2018. https://arxiv.org/abs/1804.04368
    """
    def __init__(self,
                 layer,
                 iteration=1,
                 training=True,
                 device="cuda"):
        """Initializer.

        Args:
        layer: A PyTorch nn.Conv2d layer to apply normalization to.
        iteration: (int) The number of power iteration to perform to estimate
            weight matrix's singular value.
        norm_multiplier: (float) Multiplicative constant to threshold the
            normalization. Usually under normalization, the singular value will
            converge to this value.
        training: (bool) Whether to perform power iteration to update the singular
            value estimate.
        """
    
        super().__init__()
        self.iteration = iteration
        self.do_power_iteration = training
        self.norm_multiplier = 1

        if not isinstance(layer, nn.Conv2d):
            raise ValueError('layer must be a `nn.Conv2d` instance.')

        self.layer = layer
        self.stride = layer.stride
        self.u = None
        self.v = None
        self.device = torch.device(device)

    @torch.autograd.no_grad()
    def compute_u_v(self, x):

        # Call the layer once to get shapes
        y = self.layer(x)
        in_shape = (1, *x.shape[1:])
        out_shape = (1, *y.shape[1:])
        
        # Initialise the vectors for power iteration
        self.u = nn.Parameter(torch.randn(out_shape), requires_grad=False).to(self.device)
        self.v = nn.Parameter(torch.randn(in_shape), requires_grad=False).to(self.device)

    def forward(self, x):
        if self.u is None or self.v is None:
            self.compute_u_v(x)
        
        # Only re-normalise the weight matrix during training
        if self.training:
            self.update_weights()
        
        x = self.layer(x)
        return x

    def update_weights(self, return_sigma=False):
        u_hat = self.u
        v_hat = self.v

        if self.do_power_iteration:
            for _ in range(self.iteration):
                # Updates v.
                v_ = F.conv_transpose2d(u_hat, self.layer.weight, 
                                        stride=self.stride, 
                                        padding=self.layer.padding)
                v_hat = F.normalize(v_, dim=(1,2,3))

                # Updates u.
                u_ = F.conv2d(v_hat, self.layer.weight, 
                              stride=self.stride, 
                              padding=self.layer.padding)
                u_hat = F.normalize(u_, dim=(1,2,3))

        v_w_hat = F.conv2d(v_hat, self.layer.weight, 
                           stride=self.stride, 
                           padding=self.layer.padding)

        # Normalise by the spectral norm
        sigma = torch.matmul(v_w_hat.reshape(1,-1), u_hat.reshape(-1,1))
        sigma = sigma.view([])
        w_norm = self.norm_multiplier / sigma * self.layer.weight

        # Copy new weight matrix into the CNN (and log u, v, sigma)
        self.u.data.copy_(u_hat)
        self.v.data.copy_(v_hat)
        self.layer.weight.data.copy_(w_norm)
        if return_sigma:
            return sigma
