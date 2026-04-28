import torch
import os
import pandas as pd
from torch import nn
import numpy as np
from model.pix_encoder import PixMixEncoder
from data.data_utilities import get_dataloader, DataLoader
from tqdm import tqdm
import logging

# Output location
output_dir = "out/"
model_dir = os.path.join(
    output_dir,
    "model_weights/"
)
logs_dir = os.path.join(
    output_dir,
    "logs/"
)
test_dir = os.path.join(
    output_dir,
    "test_output/"
)
for dir in [
    output_dir,
    logs_dir,
    model_dir,
    test_dir
]:
    os.makedirs(dir, exist_ok=True)

# log
logger_location = os.path.join(logs_dir, "training_log")
logging.basicConfig(filename=logger_location, level=logging.INFO)
logger = logging.getLogger(__name__)

# Training setup
epochs = 100

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = PixMixEncoder(target_t=100).to(device)
train_set, validation_set, test_set = get_dataloader(dataset_dir="dataset_jamendo/")

l1_loss_fn = nn.L1Loss()
# Loss parameters
lamda_log = 1.0
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


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(model: PixMixEncoder, dataloader: DataLoader, optimizer = optimizer, device: torch.device = device):
    model.train()
    running_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()
        input_tensor = batch["input"].to(device)
        target_tensor = batch["target"].to(device)
        output = model(input_tensor)
        loss = spectrogram_loss(output.squeeze(1), target_tensor)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_loss = running_loss / len(dataloader)

    return avg_loss

def evaluate(model: PixMixEncoder, dataloader: DataLoader, device: torch.device = device):
    model.eval()
    running_loss = 0

    for batch in dataloader:
        input_tensor = batch["input"].to(device)
        target_tensor = batch["target"].to(device)

        with torch.no_grad():
            output = model(input_tensor)
        loss = spectrogram_loss(output.squeeze(1), target_tensor)
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)

    return avg_loss

def test(model: PixMixEncoder, dataloader: DataLoader, device: torch.device = device):
    model.eval()
    running_loss = 0

    test_logs = {
        "name": [],
        "loss": []
    }

    for idx, batch in enumerate(dataloader):
        input_tensor = batch["input"].to(device)
        target_tensor = batch["target"].to(device)

        with torch.no_grad():
            output = model(input_tensor)
        loss = spectrogram_loss(output, target_tensor)

        # Log results
        test_logs["loss"].append(loss)
        test_logs["name"].append(batch["name"])

        # Update running loss
        running_loss += loss

        # Save output
        output = output.cpu().numpy()
        np.save(os.path.join(
            test_dir,
            f"{idx}_{batch["name"]}.npy"
        ))
    
    # Save logs
    pd.DataFrame(test_logs).to_csv(
        os.path.join(
            test_dir,
            "test_logs.csv"
        )
    )

    avg_loss = running_loss / len(dataloader)
    return avg_loss

best_loss = None
for epoch in tqdm(range(epochs)):
    save_model = False
    train_loss = train(model=model, dataloader=train_set)
    eval_loss = evaluate(model=model, dataloader=validation_set)
    if best_loss == None:
        best_loss = eval_loss
        save_model = True
    elif best_loss > eval_loss:
        best_loss = eval_loss
        save_model = True
    logger.info(f"Epoch: {epoch}\nTrain loss: {train_loss}\nEvaluation loss: {eval_loss}\nSave model: {save_model}")

    if save_model:
        model_file = os.path.join(
            model_dir,
            f"model_e{epoch}.pt"
        )
        torch.save(model.state_dict(), model_file)

# Test best model
best_model_dict = torch.load(model_file)
model.load_state_dict(best_model_dict)

test(model=model, dataloader=test_set)
