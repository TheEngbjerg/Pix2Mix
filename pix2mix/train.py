import torch
from diffwave.inference import predict
from torch import nn
from model.encoder.pix_encoder import PixEncoder
from data_utilities import get_dataloader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

epochs = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = PixEncoder().to(device)



def train(model: PixEncoder):
    dataloader = get_dataloader(dataset_dir="dataset_10/")
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for batch in dataloader:
        optimizer.zero_grad()
        input = batch["input"].to(device)
        target = batch["target"].to(device)
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    
    return output

for epoch in tqdm(range(epochs)):
    sample = train(model=model)

predict(sample, model_dir="diffwave-ljspeech-22kHz-1000578.pt", device=device)
