import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import wandb

from unet import Unet
from diffusion import GaussianDiffusion, DiffusionImageAPI
from data import ImageDataset

def train():
  batch_size = 8
  dataloader = torch.utils.data.DataLoader(
    ImageDataset(size=batch_size),
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
    image_size=(120, 80),
    #image_size=(80, 120),
    #image_size=(16, 16),
  )
  imageAPI = DiffusionImageAPI(diffusion)

  optimizer = optim.Adam(model.parameters(), lr=1e-4)
  criterion = nn.MSELoss()

  epochs = int(10_000)
  pbar = tqdm(total=int(epochs * len(dataloader)))
  loss_every_n_steps = 10
  image_every_n_steps = 100
  device = "cuda"

  model.to(device)
  diffusion.to(device)
  step_i = 0
  acc_loss = 0
  for epoch in range(epochs):
    for image in dataloader:
      step_i += 1 
      # (batch_size, image_width, image_height, channels)
      image = diffusion.normalize_image(image)
      t = diffusion.sample_time_steps(batch_size)

      noisy_image, noise_added_to_image = diffusion.apply_noise(image, t)

      noise_added_to_image = noise_added_to_image.to(device)
      noisy_image = noisy_image.to(device)
      t = t.to(device)

      predicted_noise_added_to_image = model(noisy_image, t)

      # we are trying to predict the noise added to images
      # thus our loss is only on the actual noise itself
      loss = criterion(predicted_noise_added_to_image, noise_added_to_image)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      acc_loss += loss.item()


      if step_i % loss_every_n_steps == 0:
        acc_loss /= loss_every_n_steps
        pbar.set_description(f"Loss: {acc_loss:.4f}")
        wandb.log({
          "epoch": epoch,
          "nr_images": batch_size * step_i,
          "train_loss": acc_loss,
        }, step=step_i)
        acc_loss = 0
      
      if step_i % image_every_n_steps == 0:
        image, _ = diffusion.sample(1, show_progress=False)
        wandb.log({
          "example_image": wandb.Image(imageAPI.tensor_to_image(image.squeeze(0))),
        }, step=step_i)
      
      pbar.update(1)

  pbar.close()
  torch.save(model.state_dict(), "./out/model.pt")

if __name__ == "__main__":
  wandb.init(
    project="movie-diffusion",
  )
  train()
