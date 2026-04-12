"""
# PixEncoder
The image encoder for Pix2Mix
"""

import torch
from torch import nn
from torchvision import models


class PixEncoder(nn.Module):
    def __init__(self, n_mels=80):
        super().__init__()
        self.n_mels = n_mels
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.projection = nn.Linear(512, n_mels)

    def forward(self, image):

        x = self.backbone(image)  # (batch, 512, 1, 1)

        # Linear expect 2D --> needs to squuze out the two 1 dimensions
        x = x.squeeze(-1).squeeze(-1)

        x = self.projection(x)

        # unsqueeze to fit with the diffwave dimensions
        x = x.unsqueeze(-1)  # (batch, n_mels, 1)

        return x
