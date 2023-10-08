import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class GaussianDiffusion:
  def __init__(self, model, noise_steps, beta_0, beta_T, image_size):
    """

    model: the model to be trained (nn.Module)
    noise_steps: the number of steps to apply noise (int)
    beta_0: the initial value of beta (float)
    beta_T: the final value of beta (float)
    image_size: the size of the image (int, int)
    """

    self.model = model
    self.noise_steps = noise_steps
    self.beta_0 = beta_0
    self.beta_T = beta_T
    self.image_size = image_size

    self.betas = self.beta_schedule()
    self.alphas = 1.0 - self.betas
    # cumulative product of alphas, so we can optimize forward process calculation
    self.alpha_hat = torch.cumprod(self.alphas, dim=0)

  def beta_schedule(self, schedule="linear"):
    if schedule == "linear":
      return torch.linspace(self.beta_0, self.beta_T, self.noise_steps)
  
  def sample_time_steps(self, batch_size=1):
    return torch.randint(0, self.noise_steps, (batch_size,))
  
  
  def q(self, x, t):
    """
    Forward process
    """
    pass
  
  def p(self, x, t):
    """
    Backward process
    """
    pass 

  
  def apply_noise(self, x, t):
    sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
    sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])
    # standard normal distribution
    epsilon = torch.randn_like(x)

    # Eq 2. in DDPM paper
    noisy_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon
    return torch.clip(noisy_image, -1.0, 1.0)
  
  @staticmethod
  def normalize_image(x):
    # normalize image to [-1, 1]
    return x / 255.0 * 2.0 - 1.0
  
  @staticmethod
  def denormalize_image(x):
    # denormalize image to [0, 255]
    return (x + 1.0) / 2.0 * 255.0

class DiffusionImageAPI:
  def __init__(self, diffusion_model):
    self.diffusion_model = diffusion_model
  
  def get_noisy_image(self, image, t):
    x = torch.tensor(np.array(image))
    x = self.diffusion_model.normalize_image(x)
    y = self.diffusion_model.apply_noise(x, t)
    y = self.diffusion_model.denormalize_image(y)
    return Image.fromarray(y.numpy().astype(np.uint8))

  
  def get_noisy_images(self, image, time_steps):
    """
    image: the image to be processed PIL.Image
    time_steps: the number of time steps to apply noise (int)
    """
    return [self.get_noisy_image(image, t) for t in time_steps]