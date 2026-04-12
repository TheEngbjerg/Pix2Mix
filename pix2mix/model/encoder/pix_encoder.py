"""
# PixEncoder
The image encoder for Pix2Mix
"""

import torch
from torch import nn
from torchvision import models


class PixEncoder(nn.Module):
    def __init__(self, n_mels=80, T=62):
        super().__init__()
        self.n_mels = n_mels
        self.T = T

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1]
        )  # last dimension (batch, 512, 1, 1)

        self.projection = nn.Linear(512, self.n_mels * T)

        # # Custom layers to map to spectrogram dimensions
        # self.upsample = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
        #     ),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(
        #         256, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        #     ),
        #     nn.Sigmoid(),  # For amplitude values between 0 and 1
        # )

    def forward(self, image):
        x = self.backbone(image)  # (batch, 512, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (batch, 512)
        x = self.projection(x)  # (batch, n_mels * 62)
        x = x.view(-1, self.n_mels, self.T)  # (batch, n_mels, 62)

        # transform normalized to [0,1] --> need same constrain
        x = torch.sigmoid(x)

        return x
