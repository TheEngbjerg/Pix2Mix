import torch
import os
from model.pix_encoder import PixMixEncoder
from model.loss_fn import spectrogram_loss
from data.data_utilities import get_dataloader, DataLoader, MAX_FRAMES
from tqdm import tqdm
import logging
from argparse import ArgumentParser

def create_parser():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-directory", type=str, required=True, help="Path to input")
    parser.add_argument("-e", "--epochs", type=int, required=False, default=100, help="Epoch amount")
    args = parser.parse_args()

    return args.input_directory, args.epochs

# log
# logger_location = os.path.join(logs_dir, "training_log")
# logging.basicConfig(filename=logger_location, level=logging.INFO)
# logger = logging.getLogger(__name__)

# Training setup

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = PixMixEncoder(target_t=MAX_FRAMES).to(device)


learning_rate = 1e-4


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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


if __name__ == "__main__":
    input_directory, epochs = create_parser()
    train_set, validation_set, test_set = get_dataloader(dataset_dir=input_directory)
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
