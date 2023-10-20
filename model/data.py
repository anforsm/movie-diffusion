import torch
import torch.nn
import numpy as np
from PIL import Image
import requests

class ImageDataset(torch.utils.data.Dataset):
  def __init__(self, size=1):
    super().__init__()
    simple_image = self.get_simple_image()
    self.images = [self.get_pil_image().copy() for _ in range(size)]
  
  def get_pil_image(self) -> Image:
    url = "https://www.themoviedb.org/t/p/w1280/6oom5QYQ2yQTMJIbnvbkBL9cHo6.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    w, h = image.size
    image.convert("RGB")
    return image
  
  def get_simple_image(self):
    url = "https://www.themoviedb.org/t/p/w1280/6oom5QYQ2yQTMJIbnvbkBL9cHo6.jpg"
    #url = "https://htmlcolorcodes.com/assets/images/colors/red-color-solid-background-1920x1080.png"
    image = Image.open(requests.get(url, stream=True).raw)
    w, h = image.size
    #image = image.resize((w // 16, h // 16))
    image = image.resize((80, 120))
    image = image.convert("RGB")
    return torch.from_numpy(np.array(image))
  
  def __getitem__(self, idx):
    return self.images[idx]
  
  def __len__(self):
    return len(self.images)