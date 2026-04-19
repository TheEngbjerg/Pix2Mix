import torch
from torch import nn
from model.encoder.pix_encoder import PixEncoder
from data_utilities import get_dataloader, DataLoader
from tqdm import tqdm
import logging

logging.basicConfig(filename='pix2mix_training.log', level=logging.INFO)
logger = logging.getLogger(__name__)

epochs = 100
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = PixEncoder().to(device)

train_set, validation_set = get_dataloader(dataset_dir="dataset_10/")
loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(model: PixEncoder, dataloader: DataLoader, loss_fn = loss_fn, optimizer = optimizer, device: torch.device = device):
    model.train()
    running_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()
        input_tensor = batch["input"].to(device)
        target_tensor = batch["target"].to(device)
        output = model(input_tensor)
        loss = loss_fn(output, target_tensor)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_loss = running_loss / len(dataloader)

    return avg_loss

def evaluate(model: PixEncoder, dataloader: DataLoader, loss_fn = loss_fn, device: torch.device = device):
    model.eval()
    running_loss = 0

    for batch in dataloader:
        input_tensor = batch["input"].to(device)
        target_tensor = batch["target"].to(device)

        with torch.no_grad():
            output = model(input_tensor)
        loss = loss_fn(output, target_tensor)
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)

    return avg_loss


for epoch in tqdm(range(epochs)):
    logger.info(f"\nEpoch: {epoch}")
    train_loss = train(model=model, dataloader=train_set, loss_fn=loss_fn)
    logger.info(f"Train loss: {train_loss}")
    eval_loss = evaluate(model=model, dataloader=validation_set, loss_fn=loss_fn)
    logger.info(f"Evaluation loss: {eval_loss}")
