import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from diffwave.preprocess import transform
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

MAX_FRAMES = 20000  # choose a fixed length that fits your data/model


def pad_or_crop_spectrogram(spectrogram, max_frames=MAX_FRAMES):
    if spectrogram.shape[1] < max_frames:
        pad = max_frames - spectrogram.shape[1]
        spectrogram = F.pad(spectrogram, (0, pad))
    else:
        spectrogram = spectrogram[:, :max_frames]
    return spectrogram


def load_image(path: str):
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = transforms.ToTensor()(image)
    return image


class JamendoDataset(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.folders = os.listdir(dataset_dir)

    def __len__(self):  # return how many samples there are
        return len(self.folders)

    def __getitem__(self, idx):  # load and return one sample by index
        folder = os.path.join(self.dataset_dir, self.folders[idx])
        cover_path = os.path.join(folder, "cover.jpg")
        audio_path = os.path.join(folder, "audio.wav")
        np_path = audio_path + ".spec.npy"

        # Load
        image = load_image(cover_path)

        if not os.path.exists(np_path):
            transform(audio_path)

        spectrogram = torch.from_numpy(np.load(np_path))
        spectrogram = pad_or_crop_spectrogram(spectrogram)

        return {"input": image, "target": spectrogram, "name": self.folders[idx]}


def get_dataloader(
    dataset_dir: str, batch_size: int = 8, test: bool = False
) -> tuple[DataLoader, DataLoader] | DataLoader:
    dataset = JamendoDataset(dataset_dir)
    if not test:
        train_dataset, validation_dataset,  = random_split(
            dataset, [0.8, 0.2]
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
        

        return train_loader, val_loader
    
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return test_loader
