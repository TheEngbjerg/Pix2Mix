"""
# PixEncoder
The image encoder for Pix2Mix
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class ConvolutionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,k, s, p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = ConvolutionBlock(in_ch + skip_ch, out_ch)
        self.conv2 = ConvolutionBlock(out_ch, out_ch)
    
    def forward(self, x, skip = None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.conv1 =nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.maxpool(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        return x1, x3, x4, x5, x6

class PixMixEncoder(nn.Module):

    def __init__(self, n_mels=80, out_channels=1, target_t = 20000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = ResNetEncoder()

        self.up1 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up3 = UpBlock(128, 64, 64)
        self.up4 = UpBlock(64, 32, 64)
        self.up5 = UpBlock(32, 16, 0)

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

        self.n_mels = n_mels
        self.target_t = target_t
    
    def forward(self, x):
        x1, x3, x4, x5, x6 = self.encoder(x)

        d1 = self.up1(x6, x5)
        d2 = self.up2(d1, x4)
        d3 = self.up3(d2, x3)
        d4 = self.up4(d3, x1)
        d5 = self.up5(d4, None)

        out = self.final_conv(d5)

        out = F.interpolate(out, size=(self.n_mels, self.target_t), mode="bilinear", align_corners=False)

        return out
