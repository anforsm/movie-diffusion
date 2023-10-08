import torch
import torch.nn
import numpy as np
from PIL import Image
import requests

class ImageDataset(torch.utils.data.Dataset):
  def __init__(self):
    super().__init__()
    self.images = [self.get_simple_image()]
  
  def get_simple_image(self):
    url = "https://www.themoviedb.org/t/p/w1280/6oom5QYQ2yQTMJIbnvbkBL9cHo6.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return torch.from_numpy(np.array(image))
  
  def __getitem__(self, idx):
    return self.images[idx]
  
  def __len__(self):
    return len(self.images)