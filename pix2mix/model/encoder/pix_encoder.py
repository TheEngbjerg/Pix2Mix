"""
# PixEncoder
The image encoder for Pix2Mix
"""
import torch
from torch import nn

class C3K2Block(nn.Module):

    def __init__(self, input_dimensions, output_dimensions, expansion = 0.5, *args, **kwargs):
        hidden_channels = int(input_dimensions * expansion)
        self.conv1 = nn.Conv2d(input_dimensions, hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, output_dimensions)
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)

class AttentionBlock(nn.Module):

    def __init__(self, in_dimensions, *args, **kwargs):
        self.query = nn.Linear(in_dimensions, in_dimensions)
        self.key = nn.Linear(in_dimensions, in_dimensions)
        self.value = nn.Linear(in_dimensions, in_dimensions)
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        self.Q: torch.Tensor = self.query(x)
        self.K: torch.Tensor = self.key(x)
        self.V: torch.Tensor = self.value(x)

        x, _ = self._scaled_dot_product()
        return x

    
    def _scaled_dot_product(self, mask = None):
        dk = self.Q.size(-1)
        scores = torch.matmul(self.Q, self.K.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        if mask != None:
            scores = scores.masked_fill(mask == 0, self.float("-inf"))
        
        attention_weights = nn.functional.softmax(scores, dim=1)

        output = torch.matmul(attention_weights, self.V)
        return output, attention_weights


class PixEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self):
        pass