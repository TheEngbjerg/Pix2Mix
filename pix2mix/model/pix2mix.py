"""
# Pix2Mix
Encode and decode wrapper module
"""
from torch import nn
from decoder.diffwave.src.diffwave.model import DiffWave

class Pix2Mix(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self):
        pass
