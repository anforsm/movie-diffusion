import torch
import numpy as np
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

if __name__ == "__main__":
  # load pt model
  model = Unet(
      dim = 64,
      dim_mults = (1, 2, 4, 8),
      flash_attn = True
  )
  diffusion = GaussianDiffusion(
      model,
      image_size = 32,
      timesteps = 200,           # number of steps
      sampling_timesteps = 200    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
  )
  trainer = Trainer(
      diffusion,
      'datasets/cifar10',
      train_batch_size = 32,
      train_lr = 1e-3,
      train_num_steps = 100,         # total training steps
      gradient_accumulate_every = 2,    # gradient accumulation steps
      ema_decay = 0.995,                # exponential moving average decay
      calculate_fid = True              # whether to calculate fid during training
  )
  trainer.load("cifar10_32x32")
  samples_images = diffusion.sample(batch_size=16)
  
  def show(img, i):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.savefig("results/sample-{}.png".format(i))

  for i in range(16):
    show(samples_images[i], i)
