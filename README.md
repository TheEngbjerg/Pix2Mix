# Pix2Mix

Pix2Mix — a simple image-to-spectrogram encoder for the Deep Learning course (2026).

This README explains how to train a model and run test/inference on new data.

## Requirements
- Python 3.8+
- PyTorch (install the version appropriate for your CUDA / CPU): https://pytorch.org/get-started/locally/
- Other Python packages: torchvision, tqdm, pillow, numpy, pandas, diffwave

Quick setup (recommended):

```powershell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install torchvision tqdm pillow numpy pandas diffwave
# Install torch from the PyTorch website for correct CUDA support
```

## Dataset layout
Each sample is a folder under your dataset directory with:
- cover.jpg   (RGB cover image)
- audio.wav   (audio file)

Example tree:

```
dataset/
  song_001/
    cover.jpg
    audio.wav
  song_002/
    cover.jpg
    audio.wav
```

The code will create spectrogram files audio.wav.spec.npy automatically using diffwave.preprocess.transform when missing.

## Training
Script: pix2mix\train.py

Usage (from repository root):

```powershell
python pix2mix\train.py -i dataset_jamendo -e 50 -l 1e-3 -b 8
```

Arguments:
- -i / --input-directory : path to dataset root (required)
- -e / --epochs          : number of epochs (default: 100)
- -l / --learning-rate   : learning rate (default: 1e-3)
- -b / --batch_size      : batch size (default: 8)

What training does:
- Creates out/ and out/model_weights/ if missing
- Saves checkpoints as out/model_weights/model_e{epoch}.pt and out/model_weights/latest.pt
- Writes logs to out/train.log
- Uses an 80/20 train/validation split
- Uses CUDA automatically if available

## Inference / Test
Script: pix2mix\inference.py

Usage:

```powershell
python pix2mix\inference.py -i dataset_test -m out/model_weights/latest.pt -b 8
```

Arguments:
- -i / --input-data     : path to dataset to run test on (required)
- -m / --model-weights  : path to model weights (default: out/model_weights/latest.pt)
- -b / --batch_size     : batch size (default: 8)

What testing does:
- Loads the model weights and runs the dataset through the model
- For each sample it saves:
  - numpy spectrogram prediction: <test_dir>\\<sample_name>\\<sample_name>.npy
  - a PNG comparing prediction & target: <test_dir>\\<sample_name>\\<sample_name>.png
- A CSV log with per-sample losses is saved as <test_dir>/logs.csv
- The test output directory name is derived from the model-weights filename (e.g. out/model_weights/latest.pt -> folder 'latest')

## Notes & tips
- MAX_FRAMES is defined in pix2mix/data/data_utilities.py (default 20000). Spectrograms will be padded or cropped to this length.
- If spectrogram generation fails, run diffwave.preprocess.transform on the audio files to create .spec.npy files manually.
- Save the exact model file used for inference for reproducibility.

## Inspecting code
Refer to pix2mix\train.py and pix2mix\inference.py for exact argument handling and behavior.
