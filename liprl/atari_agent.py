import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from torch.distributions.categorical import Categorical
from torch.nn.utils.parametrizations import spectral_norm

from liprl.networks import aol
from liprl.networks import lbdn
from liprl.networks import orthogonal as orth
from liprl.networks import scale
from liprl.networks.specnorm_conv2d import SpectralNormalizationConv2D as SpecNorm


def init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class AtariAgent(nn.Module):
    def __init__(self, 
                 n_ctrl, 
                 network="cnn", 
                 lipschitz=1.0):
        """
        Define an agent to play the Atari games.
        
        Arguments:
        n_ctrl: Number of control inputs to the environment.
        network: Network type. Choose from the following options:
            - 'cnn' (default)
            - 'spectral'
            - 'lbdn'
            - 'aol'
            - 'orthogonal'
        lipschitz: Lipschitz bound to impose on anything except CNN (default 1.0).
        """
        super().__init__()
        
        # NOTE: I've added padding to the inputs and a few maxpools due to
        # https://github.com/acfr/LBDN/issues/1. Specifically:
        #   - SandwichConv uses circular convolutions, so image size is different
        #   - SandwichConv only accepts stride lengths of 1 or 2
        #   - envpools returns image sizes of 84 x 84
        #   - If an (n x n) image has n/2 odd, then SandwichConv returns a
        #     non-square output image, hence the padding.
        # These same limitations affect the Orthogonal and AOL networks too.
        # Note also that SandwichConv and SandwichFc apply ReLU internally
        
        if network == "cnn":
            self.network = nn.Sequential(
                transforms.Pad((2,2,2,2), fill=0, padding_mode="constant"),
                init(nn.Conv2d(4, 32, 8, stride=2)), nn.ReLU(),
                nn.MaxPool2d(2),
                init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
                init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
                nn.Flatten(),
                init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
            )
            self.actor = init(nn.Linear(512, n_ctrl), std=0.01)
            self.critic = init(nn.Linear(512, 1), std=1)
            
        elif network =="spectral":
            # spectral_norm() only works for Dense layers in PyTorch
            # https://github.com/pytorch/pytorch/issues/99149
            # Instead, we use a custom version for Conv2D based on:
            # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/normalization.py
            g = np.sqrt(lipschitz)
            self.network = nn.Sequential(
                transforms.Pad((2,2,2,2), fill=0, padding_mode="constant"),
                scale.ScaleConst(g),
                SpecNorm(init(nn.Conv2d(4, 32, 8, stride=2)), iteration=5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                SpecNorm(init(nn.Conv2d(32, 64, 4, stride=2)), iteration=5),
                nn.ReLU(),
                SpecNorm(init(nn.Conv2d(64, 64, 3, stride=1)), iteration=5),
                nn.ReLU(),
                nn.Flatten(),
                spectral_norm(init(nn.Linear(64 * 7 * 7, 512)), n_power_iterations=5),
                nn.ReLU(),
            )
            self.actor = nn.Sequential(
                scale.ScaleConst(g),
                spectral_norm(init(nn.Linear(512, n_ctrl), std=0.01), n_power_iterations=5)
            )
            self.critic = init(nn.Linear(512, 1), std=1)
            
        elif network == "lbdn":
            g = np.sqrt(lipschitz)
            self.network = nn.Sequential(
                transforms.Pad((2,2,2,2), fill=0, padding_mode="constant"),
                lbdn.SandwichConv(4, 32, 8, stride=2, scale=g),
                nn.MaxPool2d(2),
                lbdn.SandwichConv(32, 64, 4, stride=2),
                lbdn.SandwichConv(64, 64, 3, stride=1),
                nn.Flatten(),
                lbdn.SandwichFc(64 * 11 * 10, 512),
            )
            self.actor = lbdn.SandwichLin(512, n_ctrl, scale=g)
            self.critic = init(nn.Linear(512, 1), std=1)
            
        elif network == "aol":
            g = np.sqrt(lipschitz)
            self.network = nn.Sequential(
                transforms.Pad((2,2,2,2), fill=0, padding_mode="constant"),
                aol.AolConvLin(4, 32, 8, scale=g), nn.ReLU(),
                nn.MaxPool2d(4), # AOL does not support stride so pool here.
                aol.AolConvLin(32, 64, 4), nn.ReLU(),
                nn.MaxPool2d(2), # AOL does not support stride so pool here.
                aol.AolConvLin(64, 64, 3), nn.ReLU(),
                nn.Flatten(),
                aol.AolLin(64 * 10 * 10, 512), nn.ReLU(),
            )
            self.actor = aol.AolLin(512, n_ctrl, scale=g)
            self.critic = init(nn.Linear(512, 1), std=1)
            
        elif network == "orthogonal":
            g = np.sqrt(lipschitz)
            self.network = nn.Sequential(
                transforms.Pad((2,2,2,2), fill=0, padding_mode="constant"),
                orth.OrthogonConvLin(4, 32, 8, stride=2, scale=g), nn.ReLU(),
                nn.MaxPool2d(2),
                orth.OrthogonConvLin(32, 64, 4, stride=2), nn.ReLU(),
                orth.OrthogonConvLin(64, 64, 3, stride=1), nn.ReLU(),
                nn.Flatten(),
                orth.OrthogonLin(64 * 11 * 10, 512), nn.ReLU(),
            )
            self.actor = orth.OrthogonLin(512, n_ctrl, scale=g)
            self.critic = init(nn.Linear(512, 1), std=1)
            
        else:
            raise ValueError(f"Unrecognised network type '{network}'")

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))
    
    def get_logits(self, x):
        return self.actor(self.network(x / 255.0))
    
    def get_action(self, x):
        logits = self.actor(self.network(x / 255.0))
        probs = Categorical(logits=logits)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        value = self.critic(hidden)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value
