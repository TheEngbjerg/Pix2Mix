"""
# PixEncoder
The image encoder for Pix2Mix
"""
from torch import nn

class PixEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self):
        pass