import torch
from torch import nn
from model.encoder.pix_encoder import PixEncoder
from data_utilities import get_dataloader, DataLoader
from tqdm import tqdm

epochs = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = PixEncoder().to(device)

train_set, validation_set = get_dataloader(dataset_dir="dataset_10/")
loss_fn = nn.MSELoss()

def train(model: PixEncoder, dataloader: DataLoader, loss_fn):
    running_loss = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for batch in dataloader:
        optimizer.zero_grad()
        input_tensor = batch["input"].to(device)
        target_tensor = batch["target"].to(device)
        output = model(input_tensor)
        loss = loss_fn(output, target_tensor)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    print(f"Train loss: {running_loss / len(dataloader.dataset)}")
    return output

def evaluate(model: PixEncoder, dataloader: DataLoader, loss_fn):
    running_loss = 0

    for batch in dataloader:
        input_tensor = batch["input"]
        target_tensor = batch["target"]

        with torch.no_grad():
            output = model(input_tensor)
        loss = loss_fn(output, target_tensor)
        running_loss += loss.item()
    
    print(f"Evaluation loss: {running_loss / len(dataloader.dataset)}")
    return output


for epoch in tqdm(range(epochs)):
    train(model=model, dataloader=train_set, loss_fn=loss_fn)
    evaluate(model=model, dataloader=validation_set, loss_fn=loss_fn)
