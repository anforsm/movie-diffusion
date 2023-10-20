import warnings
warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created")

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

from unet import Unet
from diffusion import GaussianDiffusion, DiffusionImageAPI
from data import ImageDataset

LOG_WANDB = True

IMAGE_WIDTH = 32 
IMAGE_HEIGHT = 32

BATCH_SIZE = 12
DEVICE = "cuda"

if LOG_WANDB:
  import wandb

image_transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Lambda(lambda x: x * 2 - 1),
  transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), antialias=True),
])
# 192 x 128
# 96 x 64
# 48 x 32
# 24 x 16
# 12 x 8
# 6 x 4
# 3 x 2

reverse_transform = transforms.Compose([
  transforms.Lambda(lambda x: (x + 1) / 2),
  transforms.ToPILImage(),
])

def collate_fn(batch):
  #return torch.stack([image_transform(image["image"]) for image in batch])
  return torch.stack([image_transform(image["img"]) for image in batch])

def train():
  batch_size = BATCH_SIZE
  #dataset = ImageDataset(size=batch_size*2),
  #dataset = load_dataset("skvarre/movie_posters", split="train")
  dataset = load_dataset("cifar10", split="train")
  dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
  ) 

  model = Unet(
    image_channels=3,
  )
  print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

  diffusion = GaussianDiffusion(
    model=model,
    noise_steps=256,
    #noise_steps=1024,
    beta_0=1e-4,
    beta_T=0.02,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    #image_size=(80, 120),
    #image_size=(16, 16),
  )
  imageAPI = DiffusionImageAPI(diffusion)

  optimizer = optim.Adam(model.parameters(), lr=1e-4)
  criterion = nn.MSELoss()

  epochs = int(50000)
  pbar = tqdm(total=int(epochs * len(dataloader)))
  loss_every_n_steps = 10
  image_every_n_steps = 500 
  device = DEVICE

  model.to(device)
  diffusion.to(device)
  step_i = 0
  acc_loss = 0
  for epoch in range(epochs):
    for image in dataloader:
      step_i += 1 
      # (batch_size, image_width, image_height, channels)
      #image = diffusion.normalize_image(image)
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

        if LOG_WANDB:
          wandb.log({
            "epoch": epoch,
            "nr_images": batch_size * step_i,
            "train_loss": acc_loss,
          }, step=step_i)

        acc_loss = 0
      
      if step_i % image_every_n_steps == 0:
        image, _ = diffusion.sample(1, show_progress=False)
        if LOG_WANDB:
          wandb.log({
            "example_image": wandb.Image(imageAPI.tensor_to_image(image.squeeze(0).permute(1,2,0))),
          }, step=step_i)
      
      pbar.update(1)

  pbar.close()
  torch.save(model.state_dict(), "./out/model.pt")

if __name__ == "__main__":
  if LOG_WANDB:
    wandb.init(
      project="movie-diffusion",
    )
  train()
