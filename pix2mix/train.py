import torch
import os
from torch import nn
from model.pix_encoder import PixMixEncoder
from data.data_utilities import get_dataloader, DataLoader
from tqdm import tqdm
import logging

# Output location
output_dir = "out/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# log
logger_location = os.path.join(output_dir, "training_log")
logging.basicConfig(filename=logger_location, level=logging.INFO)
logger = logging.getLogger(__name__)

# Training setup
epochs = 100

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = PixMixEncoder(target_t=100).to(device)
train_set, validation_set = get_dataloader(dataset_dir="dataset_10/")

loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(model: PixMixEncoder, dataloader: DataLoader, loss_fn = loss_fn, optimizer = optimizer, device: torch.device = device):
    model.train()
    running_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()
        input_tensor = batch["input"].to(device)
        target_tensor = batch["target"].to(device)
        output = model(input_tensor)
        loss = loss_fn(output.squeeze(1), target_tensor)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_loss = running_loss / len(dataloader)

    return avg_loss

def evaluate(model: PixMixEncoder, dataloader: DataLoader, loss_fn = loss_fn, device: torch.device = device):
    model.eval()
    running_loss = 0

    for batch in dataloader:
        input_tensor = batch["input"].to(device)
        target_tensor = batch["target"].to(device)

        with torch.no_grad():
            output = model(input_tensor)
        loss = loss_fn(output.squeeze(1), target_tensor)
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)

    return avg_loss

best_loss = None
for epoch in tqdm(range(epochs)):
    save_model = False
    train_loss = train(model=model, dataloader=train_set, loss_fn=loss_fn)
    eval_loss = evaluate(model=model, dataloader=validation_set, loss_fn=loss_fn)
    if best_loss == None:
        best_loss = eval_loss
        save_model = True
    elif best_loss > eval_loss:
        best_loss = eval_loss
        save_model = True
    logger.info(f"Epoch: {epoch}\nTrain loss: {train_loss}\nEvaluation loss: {eval_loss}\nSave model: {save_model}")

    if save_model:
        model_file = os.path.join(
            output_dir,
            f"model_e{epoch}.pt"
        )
        torch.save(model.state_dict(), model_file)
