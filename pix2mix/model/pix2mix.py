"""
# Pix2Mix
Encode and decode wrapper module
"""
from torch import nn
from decoder.diffwave.src.diffwave.model import DiffWave
from encoder.pix_encoder import PixEncoder

class Pix2Mix(nn.Module):

    def __init__(self, *args, **kwargs):
        self.encoder = PixEncoder()
        self.decoder = DiffWave()
        super().__init__(*args, **kwargs)
    
    def forward(self):
        pass
