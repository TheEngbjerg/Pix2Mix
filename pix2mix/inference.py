import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

from model.loss_fn import spectrogram_loss
from model.pix_encoder import PixMixEncoder
from data.data_utilities import DataLoader, get_dataloader, MAX_FRAMES
from utils.directory_helpers import create_test, get_test_directory
from utils.plot_helpers import create_pref_target_spectrogram

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-data", type=str, required=True, help="Input data")
    parser.add_argument("-m", "--model-weights", type=str, required=False, default="out/model_weights/latest.pt", help="Path to model weights")
    args = parser.parse_args()

    return args.input_data, args.model_weights

def test(experiment_name: str, model: PixMixEncoder, dataloader: DataLoader, device: torch.device):
    test_dir, log_path = create_test(experiment_name)
    model.eval()
    running_loss = 0

    test_logs = {
        "name": [],
        "loss": []
    }

    for idx, batch in tqdm(enumerate(dataloader), desc="Running test"):
        input_tensor = batch["input"].to(device)
        target_tensor = batch["target"].to(device)

        with torch.no_grad():
            output = model(input_tensor)
        output = output.squeeze(1)
        loss = spectrogram_loss(output, target_tensor)
        running_loss += loss.item()

        # Per sample processing
        output_np = output.cpu().numpy()
        target_np = target_tensor.cpu().numpy()
        batch_size = output.shape[0]

        for i in range(batch_size):
            name = batch["name"][i]
            pred = output_np[i]
            targ = target_np[i]
            loss_i = spectrogram_loss(torch.from_numpy(pred).unsqueeze(0), torch.from_numpy(targ).unsqueeze(0)).item()

            test_logs["name"].append(name)
            test_logs["loss"].append(loss_i)

            

            np_path, figure_path = get_test_directory(name=name, test_path=test_dir)
            create_pref_target_spectrogram(pred, targ, save_path=figure_path, title_prefix=name)

            np.save(np_path, pred)
    
    # Save logs
    pd.DataFrame(test_logs).to_csv(log_path)

    avg_loss = running_loss / len(dataloader)
    return avg_loss


if __name__ == "__main__":
    input_path, model_weights = get_parser()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = PixMixEncoder(target_t=MAX_FRAMES).to(device)

    state_dict = torch.load(model_weights)
    model.load_state_dict(state_dict)

    dataloader = get_dataloader(dataset_dir=input_path, test=True)

    test_loss = test(
        experiment_name=model_weights,
        model=model,
        dataloader=dataloader,
        device=device
    )

    print(f"Test loss: {test_loss}")