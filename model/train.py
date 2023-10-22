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

from conf import LOG_WANDB, IMAGE_WIDTH, IMAGE_HEIGHT, BATCH_SIZE, DEVICE, HF_TRAIN_DATASET, HF_VAL_DATASET, VAL_EVERY_N_STEPS, IMAGE_EVERY_N_STEPS, EPOCHS, HF_IMAGE_KEY, HF_TRAIN_SPLIT, HF_VAL_SPLIT, BETA_SCHEDULE, NOISE_STEPS

if LOG_WANDB:
  import wandb

image_transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Lambda(lambda x: x * 2 - 1),
  transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), antialias=True),
])

reverse_transform = transforms.Compose([
  transforms.Lambda(lambda x: (x + 1) / 2),
  transforms.ToPILImage(),
])

def collate_fn(batch):
  processed_images = []
  for image in batch:
      img = image_transform(image[HF_IMAGE_KEY])
      if img.shape[0] == 1:  # Check if the image is grayscale
          img = img.repeat(3, 1, 1)  # Convert to RGB by repeating the single channel
      processed_images.append(img)
  
  return torch.stack(processed_images)
  #return torch.stack([image_transform(image[HF_IMAGE_KEY]) for image in batch])

def train():
  batch_size = BATCH_SIZE
  dataset = load_dataset(HF_TRAIN_DATASET, split=HF_TRAIN_SPLIT)
  dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
  ) 
  val_dataset = load_dataset(HF_VAL_DATASET, split=HF_VAL_SPLIT)
  val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
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
    noise_steps=NOISE_STEPS,
    #noise_steps=1024,
    beta_0=1e-4,
    beta_T=0.02,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    #image_size=(80, 120),
    #image_size=(16, 16),
    schedule=BETA_SCHEDULE,
  )
  imageAPI = DiffusionImageAPI(diffusion)

  optimizer = optim.Adam(model.parameters(), lr=1e-4)
  criterion = nn.MSELoss()

  epochs = int(EPOCHS)
  pbar = tqdm(total=int(epochs * len(dataloader)))
  loss_every_n_steps = 10
  val_every_n_steps = VAL_EVERY_N_STEPS
  image_every_n_steps = IMAGE_EVERY_N_STEPS
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
        
      if step_i % val_every_n_steps == 0:
        val_loss = diffusion.validate(val_dataloader)
        if LOG_WANDB:
          wandb.log({
            "val_loss": val_loss,
          }, step=step_i)
      
      pbar.update(1)

  pbar.close()
  torch.save(model.state_dict(), "./out/model.pt")

if __name__ == "__main__":
  if LOG_WANDB:
    wandb.init(
      project="movie-diffusion",
      config={
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "hf_train_dataset": HF_TRAIN_DATASET,
        "hf_train_split": HF_TRAIN_SPLIT,
        "hf_val_dataset": HF_VAL_DATASET,
        "hf_val_split": HF_VAL_SPLIT,
        "hf_image_key": HF_IMAGE_KEY,
        "epochs": EPOCHS,
        "val_every_n_steps": VAL_EVERY_N_STEPS,
        "image_every_n_steps": IMAGE_EVERY_N_STEPS,
        "beta_schedule": BETA_SCHEDULE,
      }
    )
  train()
