import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from unet import Unet
from diffusion import GaussianDiffusion
from data import ImageDataset

def train():
  dataloader = torch.utils.data.DataLoader(
    ImageDataset(),
    batch_size=1,
    shuffle=True,
  ) 

  model = Unet(
    image_channels=3,
  )

  diffusion = GaussianDiffusion(
    model=model,
    noise_steps=1000,
    beta_0=1e-4,
    beta_T=0.02,
    image_size=(256, 256),
  )

  optimizer = optim.Adam(model.parameters(), lr=1e-4)
  criterion = nn.MSELoss()

  epochs = 10
  batch_size = 1
  pbar = tqdm(range(epochs))
  for epoch in pbar:
    for image in dataloader:
      image = diffusion.normalize_image(image)
      t = diffusion.sample_time_steps(batch_size)
      noisy_image = diffusion.apply_noise(image, t)
      predicted_noise = model(noisy_image, t)
      exit()

      loss = criterion(predicted_noise, noisy_image)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      pbar.set_description(f"Loss: {loss.item():.4f}")

if __name__ == "__main__":
  train()