import torch
import logging
from tqdm import tqdm
from argparse import ArgumentParser

from model.pix_encoder import PixMixEncoder
from model.loss_fn import spectrogram_loss
from utils.directory_helpers import train_setup, get_modelfile
from data.data_utilities import get_dataloader, DataLoader, MAX_FRAMES

def create_parser():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-directory", type=str, required=True, help="Path to input")
    parser.add_argument("-e", "--epochs", type=int, required=False, default=100, help="Epoch amount")
    parser.add_argument("-l", "--learning-rate", type=float, required=False, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    return args.input_directory, args.epochs, args.learning_rate

# Training setup
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(model: PixMixEncoder, dataloader: DataLoader, optimizer, device: torch.device = device):
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
    input_directory, epochs, learning_rate = create_parser()
    model_dir, logfile_location = train_setup()
    
    # log
    logging.basicConfig(filename=logfile_location, level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Training
    model = PixMixEncoder(target_t=MAX_FRAMES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_set, validation_set = get_dataloader(dataset_dir=input_directory)
    best_loss = None
    for epoch in tqdm(range(epochs)):
        save_model = False
        train_loss = train(model=model, optimizer=optimizer, dataloader=train_set)
        eval_loss = evaluate(model=model, dataloader=validation_set)
        if best_loss == None:
            best_loss = eval_loss
            save_model = True
        elif best_loss > eval_loss:
            best_loss = eval_loss
            save_model = True
        logger.info(f"Epoch: {epoch}\nTrain loss: {train_loss}\nEvaluation loss: {eval_loss}\nSave model: {save_model}")

        if save_model:
            model_file = get_modelfile(f"model_e{epoch}", model_dir)
            torch.save(model.state_dict(), model_file)
    
    model_file = get_modelfile("latest", model_dir)
    torch.save(model.state_dict(), model_file)
