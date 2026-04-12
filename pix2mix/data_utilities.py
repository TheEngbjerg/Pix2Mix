import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchaudio


class JamendoDataset(Dataset):
    def __init__(self, dataset_dir="/ceph/project/pix_audio/data/dataset_jamendo"):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.folders = [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, f))
        ]

    def __len__(self):  # return how many samples there are
        return len(self.folders)

    def __getitem__(self, idx):  # load and return one sample by index
        folder = self.folders[idx]
        cover_path = os.path.join(folder, "cover.jpg")
        audio_path = os.path.join(folder, "audio.wav")

        # Load
        image = Image.open(cover_path)
        image = transforms.ToTensor()(image)  # transform to tensor

        audio, sr = torchaudio.load(audio_path)

        return image, audio


def get_dataloader(dataset_dir, batch_size=16):  # just 16 for now
    dataset = JamendoDataset(dataset_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
