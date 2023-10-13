import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from unet import Unet
from diffusion import GaussianDiffusion, DiffusionImageAPI
from data import ImageDataset

def train():
  batch_size = 1
  dataloader = torch.utils.data.DataLoader(
    ImageDataset(),
    batch_size=batch_size,
    shuffle=False,
  ) 

  model = Unet(
    image_channels=3,
  )
  print(sum(p.numel() for p in model.parameters()))

  diffusion = GaussianDiffusion(
    model=model,
    #noise_steps=256,
    noise_steps=1024,
    beta_0=1e-4,
    beta_T=0.02,
    #image_size=(120, 80),
    image_size=(80, 120),
    #image_size=(16, 16),
  )
  diffusionAPI = DiffusionImageAPI(diffusion)

  optimizer = optim.Adam(model.parameters(), lr=1e-4)
  criterion = nn.MSELoss()

  epochs = int(5000)
  pbar = tqdm(range(epochs))
  device = "cpu"
  model.to(device)
  epoch_test = 50
  for epoch in pbar:
    acc_loss = 0.0
    for image in dataloader:
      image = diffusion.normalize_image(image)
      t = diffusion.sample_time_steps(batch_size)
      if epoch == epoch_test:
        t = torch.tensor([255])
      noisy_image = diffusion.apply_noise(image, t)

      noisy_image = noisy_image.to(device)
      t = t.to(device)
      predicted_noise = model(noisy_image, t)


      if epoch == epoch_test:
        print(t)
        print(noisy_image.shape)
        diffusionAPI.tensor_to_image(diffusion.denormalize_image(noisy_image.squeeze(0))).save("noisy_image.png")
        print(predicted_noise.shape)
        copy = predicted_noise.clone().detach().cpu()
        diffusionAPI.tensor_to_image(diffusion.denormalize_image(copy.squeeze(0))).save("predicted_image.png")


      loss = criterion(predicted_noise, noisy_image)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      acc_loss += loss.item()

    acc_loss /= len(dataloader)
    pbar.set_description(f"Loss: {acc_loss:.4f}")
  torch.save(model.state_dict(), "model.pt")

if __name__ == "__main__":
  train()