import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from unet import Unet
from diffusion import GaussianDiffusion, DiffusionImageAPI
from data import ImageDataset

def inference():
  model = Unet(
    image_channels=3,
  )
  model.load_state_dict(torch.load("./out/model.pt"))

  diffusion = GaussianDiffusion(
    model=model,
    #noise_steps=256,
    noise_steps=1024,
    beta_0=1e-4,
    beta_T=0.02,
    image_size=(120, 80),
    #image_size=(16, 16),
  )

  imageAPI = DiffusionImageAPI(diffusion)

  images, versions = diffusion.sample(1)
  #if not isinstance(images, list):
  #  print(images.shape)
  #  images = [images]
  images = []
  for image in versions:
    images.append(imageAPI.tensor_to_image(image.squeeze(0)))
  
  print(len(images))
  print(images[0])
  # make gif out of pillow images
  images[0].save('./gif_output/versions.gif',
                 save_all=True,
                 append_images=images[1:],
                 duration=100,
                 loop=0)

if __name__ == "__main__":
  inference()
