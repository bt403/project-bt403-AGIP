import torch
import os
from PIL import Image
import numpy as np
import torchvision as tv
import random as rand
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def addNoise(x, device): 
    """
    We will use this helper function to add noise to some data. 
    x: the data we want to add noise to
    device: the CPU or GPU that the input is located on. 
    """
    noiseLevel = rand.choice([0, 0.1, 0.3, 0.5])
    #return x + normal.sample(sample_shape=torch.Size(x.shape)).to(device)
    return x + (torch.randn(x.shape) * noiseLevel)

class NoisyDataset(torch.utils.data.Dataset):
  def __init__(self, in_path, mode='train', img_size=(320, 320), sigma=30):
    super(NoisyDataset, self).__init__()

    self.mode = mode #train or test
    self.in_path = in_path # ./BSDS300/images
    self.img_size = img_size # (180, 180)

    self.img_dir = os.path.join(in_path, mode)
    self.imgs = os.listdir(self.img_dir)
    self.sigma = sigma

  def __len__(self):
      return len(self.imgs)
  
  def __repr__(self):
      return "Dataset Parameters: mode={}, img_size={}, sigma={}".format(self.mode, self.img_size, self.sigma)
    
  def __getitem__(self, idx):

      img_path = os.path.join(self.img_dir, self.imgs[idx])
      clean_img = Image.open(img_path).convert('RGB')
      left = np.random.randint(clean_img.size[0] - self.img_size[0])
      top = np.random.randint(clean_img.size[1] - self.img_size[1])
      # .crop(left, upper, right, lower)
      cropped_clean = clean_img.crop([left, top, left+self.img_size[0], top+self.img_size[1]])
      transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      ground_truth = transform(cropped_clean)
      noisy = addNoise(ground_truth, device)
      return noisy, ground_truth

  def named_parameters(self, recurse=True):
        nps = nn.Module.named_parameters(self)
        for name, param in nps:
            if not param.requires_grad:
                continue
            yield name, param

class NoisyDatasetUnsup(torch.utils.data.Dataset):
  def __init__(self, in_path, mode='train', img_size=(320, 320), sigma=30):
    super(NoisyDatasetUnsup, self).__init__()

    self.mode = mode #train or test
    self.in_path = in_path # ./BSDS300/images
    self.img_size = img_size # (180, 180)

    self.img_dir = os.path.join(in_path, "Data")
    self.imgs = os.listdir(self.img_dir)
    i = 0
    for n in self.imgs:
      self.imgs[i] = os.path.join(n, "GT_SRGB_010.PNG")
      i = i+1
    self.sigma = sigma
    
  def __len__(self):
      return len(self.imgs)
  
  def __repr__(self):
      return "Dataset Parameters: mode={}, img_size={}, sigma={}".format(self.mode, self.img_size, self.sigma)
    
  def __getitem__(self, idx):

      img_path = os.path.join(self.img_dir, self.imgs[idx])
      clean_img = Image.open(img_path).convert('RGB')
      left = np.random.randint(clean_img.size[0] - self.img_size[0])
      top = np.random.randint(clean_img.size[1] - self.img_size[1])
      # .crop(left, upper, right, lower)
      cropped_clean = clean_img.crop([left, top, left+self.img_size[0], top+self.img_size[1]])
      transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      ground_truth = transform(cropped_clean)
      noisy = addNoise(ground_truth, device)
      return noisy, ground_truth

  def named_parameters(self, recurse=True):
        nps = nn.Module.named_parameters(self)
        for name, param in nps:
            if not param.requires_grad:
                continue
            yield name, param