import os
import numpy as np
from torchvision import transforms
from PIL import Image
import torchaudio
from model.decoder.diffwave.src.diffwave.preprocess import transform
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold


class JamendoDataset(Dataset):
    def __init__(self, dataset_dir):
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
        np_path = audio_path + ".spec.npy"

        # Load
        image = Image.open(cover_path)
        image = transforms.ToTensor()(image)  # transform to tensor

        if not os.path.exists(np_path):
            transform(audio_path)

        spectrogram = torch.from_numpy(np.load(np_path))

        return image, spectrogram


def get_dataloader(
    dataset_dir="/ceph/project/pix_audio/data/dataset_jamendo", batch_size=16, k=5
):  # just 16 for now
    dataset = JamendoDataset(dataset_dir)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    folds = []
    for train_indices, val_indices in kf.split(dataset):
        train_loader = DataLoader(
            Subset(dataset, train_indices), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            Subset(dataset, val_indices), batch_size=batch_size, shuffle=False
        )
        folds.append((train_loader, val_loader))

    return folds
