import torch
from model.pix_encoder import PixMixEncoder
from data.data_utilities import load_image
import numpy as np


device = torch.device("cuda")

modelfile = "out/model_e88.pt"
image = "946_Emptiness/cover.jpg"

model = PixMixEncoder()
state_dict = torch.load(modelfile)
model.load_state_dict(state_dict)
model.to(device)

image = load_image(image).to(device).unsqueeze(0)

with torch.no_grad():
    model_output = model(image).squeeze()

model_output = model_output.cpu().numpy()

np.save("output.npy", model_output)