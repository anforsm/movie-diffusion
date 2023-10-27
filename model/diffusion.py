import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
import math 
class GaussianDiffusion:
  def __init__(self, model, noise_steps, beta_0, beta_T, image_size, channels=3, schedule="linear"):
    """
    suggested betas for:
      * linear schedule: 1e-4, 0.02

    model: the model to be trained (nn.Module)
    noise_steps: the number of steps to apply noise (int)
    beta_0: the initial value of beta (float)
    beta_T: the final value of beta (float)
    image_size: the size of the image (int, int)
    """
    self.device = 'cpu'
    self.channels = channels

    self.model = model
    self.noise_steps = noise_steps
    self.beta_0 = beta_0
    self.beta_T = beta_T
    self.image_size = image_size

    self.betas = self.beta_schedule(schedule=schedule)
    self.alphas = 1.0 - self.betas
    # cumulative product of alphas, so we can optimize forward process calculation
    self.alpha_hat = torch.cumprod(self.alphas, dim=0)

  def beta_schedule(self, schedule="cosine"):
    if schedule == "linear":
      return torch.linspace(self.beta_0, self.beta_T, self.noise_steps).to(self.device)
    elif schedule == "cosine":
        return self.betas_for_alpha_bar(
            self.noise_steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
      

  def betas_for_alpha_bar(self,num_diffusion_timesteps, alpha_bar, max_beta=0.999):

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas).to(self.device)
  

  def sample_time_steps(self, batch_size=1):
    return torch.randint(0, self.noise_steps, (batch_size,)).to(self.device)
  
  def to(self,device):
    self.device = device
    self.betas = self.betas.to(device)
    self.alphas = self.alphas.to(device)
    self.alpha_hat = self.alpha_hat.to(device)

  
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
    # force x to be (batch_size, image_width, image_height, channels)
    if len(x.shape) == 3:
      x = x.unsqueeze(0)
    if type(t) == int:
      t = torch.tensor([t])
    #print(f'Shape -> {x.shape}, len -> {len(x.shape)}')
    sqrt_alpha_hat = torch.sqrt(torch.tensor([self.alpha_hat[t_] for t_ in t]).to(self.device))
    sqrt_one_minus_alpha_hat = torch.sqrt(torch.tensor([1.0 - self.alpha_hat[t_] for t_ in t]).to(self.device))
    # standard normal distribution
    epsilon = torch.randn_like(x).to(self.device)

    # Eq 2. in DDPM paper
    #noisy_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon

    """print(f'''
              Shape of x {x.shape}
              Shape of sqrt {sqrt_one_minus_alpha_hat.shape}''')"""
    
    try:
      #print(x.shape)
      #noisy_image = torch.einsum("b,bwhc->bwhc", sqrt_alpha_hat, x.to(self.device)) + torch.einsum("b,bwhc->bwhc", sqrt_one_minus_alpha_hat, epsilon)
      noisy_image = torch.einsum("b,bcwh->bcwh", sqrt_alpha_hat, x.to(self.device)) + torch.einsum("b,bcwh->bcwh", sqrt_one_minus_alpha_hat, epsilon)
    except:
      print(f'Failed image: shape {x.shape}')
      
    
    #print(f'Noisy image -> {noisy_image.shape}')
    # returning noisy iamge and the noise which was added to the image
    #return noisy_image, epsilon
    #return torch.clip(noisy_image, -1.0, 1.0), epsilon
    return noisy_image, epsilon
  
  @staticmethod
  def normalize_image(x):
    # normalize image to [-1, 1]
    return x / 255.0 * 2.0 - 1.0
  
  @staticmethod
  def denormalize_image(x):
    # denormalize image to [0, 255]
    return (x + 1.0) / 2.0 * 255.0
  
  def sample_step(self, x, t):
    device = x.device
    z = torch.randn_like(x) if t >= 1 else torch.zeros_like(x)
    z = z.to(device)
    alpha = self.alphas[t]
    print(alpha.shape)
    print(x.shape)
    one_over_sqrt_alpha = 1.0 / torch.sqrt(alpha)
    one_minus_alpha = 1.0 - alpha

    sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])
    beta_hat = (1 - self.alpha_hat[t-1]) / (1 - self.alpha_hat[t]) * self.betas[t]
    beta = self.betas[t]
    # should we reshape the params to (batch_size, 1, 1, 1) ?
    

    # we can either use beta_hat or beta_t
    # std = torch.sqrt(beta_hat)
    std = torch.sqrt(beta)
    # mean + variance * z
    predicted_noise = self.model(x, torch.tensor([t]).to(device))
    mean = one_over_sqrt_alpha * (x - one_minus_alpha / sqrt_one_minus_alpha_hat * predicted_noise)
    x_t_minus_1 = mean + std * z

    return x_t_minus_1
  
  def sample(self, num_samples, show_progress=True):
    """
    Sample from the model
    """
    self.model.eval()
    image_versions = []
    with torch.no_grad():
      x = torch.randn(1, self.channels, *self.image_size).to(self.device)
      it = reversed(range(1, self.noise_steps))
      if show_progress:
        it = tqdm(it)
      for t in it:
        image_versions.append(self.denormalize_image(torch.clip(x, -1, 1)).clone().squeeze(0))
        x = self.sample_step(x, t)
    self.model.train()
    x = torch.clip(x, -1.0, 1.0)
    return self.denormalize_image(x), image_versions
  
  def validate(self, dataloader):
    """
    Calculate the loss on the validation set
    """
    self.model.eval()
    acc_loss = 0
    for batch in dataloader:
      t = self.sample_time_steps(batch_size=batch.shape[0])
      noisy_image, added_noise = self.apply_noise(batch, t)
      noisy_image = noisy_image.to(self.device)
      added_noise = added_noise.to(self.device)
      predicted_noise = self.model(noisy_image, t)
      loss = nn.MSELoss()(predicted_noise, added_noise)
      acc_loss += loss.item()
    self.model.train()
    return acc_loss / len(dataloader)

class DiffusionImageAPI:
  def __init__(self, diffusion_model):
    self.diffusion_model = diffusion_model
  
  def get_noisy_image(self, image, t):
    x = torch.tensor(np.array(image))
    
    x = self.diffusion_model.normalize_image(x)

    y, _ = self.diffusion_model.apply_noise(x, t)
    
    y = self.diffusion_model.denormalize_image(y)
    #print(f"Shape of Image: {y.shape}")

    return Image.fromarray(y.squeeze(0).numpy().astype(np.uint8))

  
  def get_noisy_images(self, image, time_steps):
    """
    image: the image to be processed PIL.Image
    time_steps: the number of time steps to apply noise (int)
    """

    return [self.get_noisy_image(image, int(t)) for t in time_steps]
  
  def tensor_to_image(self, tensor):
    return Image.fromarray(tensor.cpu().numpy().astype(np.uint8))
