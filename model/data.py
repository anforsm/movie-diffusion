import torch
import torch.nn
import numpy as np
from PIL import Image
import requests

class ImageDataset(torch.utils.data.Dataset):
  def __init__(self, size=1):
    super().__init__()
    image_urls = [
      "https://www.themoviedb.org/t/p/w1280/6oom5QYQ2yQTMJIbnvbkBL9cHo6.jpg",
      "https://www.themoviedb.org/t/p/w1280/7WsyChQLEftFiDOVTGkv3hFpyyt.jpg",
      "https://www.themoviedb.org/t/p/w1280/wqnLdwVXoBjKibFRR5U3y0aDUhs.jpg",
      "https://www.themoviedb.org/t/p/w1280/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg",
      "https://www.themoviedb.org/t/p/w1280/iuFNMS8U5cb6xfzi51Dbkovj7vM.jpg",
      "https://www.themoviedb.org/t/p/w1280/8j58iEBw9pOXFD2L0nt0ZXeHviB.jpg",
      "https://www.themoviedb.org/t/p/w1280/h7Lcio0c9ohxPhSZg42eTlKIVVY.jpg",
      "https://www.themoviedb.org/t/p/w1280/9wSbe4CwObACCQvaUVhWQyLR5Vz.jpg",
    ]
    pil_images = [self.get_pil_image(url) for url in image_urls]
    repeats = size // len(pil_images)
    self.images = [image.copy() for _ in range(repeats) for image in pil_images]
  
  def get_pil_image(self, url) -> Image:
    image = Image.open(requests.get(url, stream=True).raw)
    w, h = image.size
    image.convert("RGB")
    return image
  
  def __getitem__(self, idx):
    return self.images[idx]
  
  def __len__(self):
    return len(self.images)