import torch.nn as nn
import torch

l1_loss_fn = nn.L1Loss()
# Loss parameters
lamda_log = 0.1
eps = 1e-5
def spectrogram_loss(prediction: torch.Tensor, target: torch.Tensor):
    prediction = torch.clamp(prediction, min=eps)
    target = torch.clamp(target, min=eps)

    l1_loss = l1_loss_fn(prediction, target)
    l1_log_loss = l1_loss_fn(
        torch.log(prediction),
        torch.log(target)
    )

    return l1_loss + lamda_log * l1_log_loss