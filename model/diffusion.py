import torch.nn as nn

class Diffusion(nn.Module):
    def __init__(self):
        super(Diffusion).__init__()

    def forward(self, x):
        raise NotImplementedError