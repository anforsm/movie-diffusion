from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

if __name__ == "__main__":
  model = Unet(
      dim = 64,
      dim_mults = (1, 2, 4, 8),
      flash_attn = True
  )

  diffusion = GaussianDiffusion(
      model,
      #image_size = 128,
      image_size = 32,
      #timesteps = 1000,           # number of steps
      timesteps = 200,           # number of steps
      #sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
      sampling_timesteps = 50    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
  )

  trainer = Trainer(
      diffusion,
      'datasets/cifar10',
      train_batch_size = 32,
      train_lr = 8e-5,
      #train_lr = 1e-3,
      #train_num_steps = 700000,         # total training steps
      train_num_steps = 999,         # total training steps
      gradient_accumulate_every = 2,    # gradient accumulation steps
      ema_decay = 0.995,                # exponential moving average decay
      #amp = True,                       # turn on mixed precision
      calculate_fid = True              # whether to calculate fid during training
  )

  print("Number of parameters:", sum(p.numel() for p in model.parameters()))
  trainer.train()
  trainer.save("cifar10_32x32")