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
    noiseLevel = rand.choice([0.5])
    #noiseLevel = rand.choice([0, 0.1, 0.3, 0.5])
    #return x + normal.sample(sample_shape=torch.Size(x.shape)).to(device)
    return x + (torch.randn(x.shape) * noiseLevel)

class NoisyDataset(torch.utils.data.Dataset):
  def __init__(self, in_path_data_1, in_path_data_2, in_path_data_3, mode='train', img_size=(320, 320), sigma=30):
    super(NoisyDataset, self).__init__()

    self.mode = mode #train or test
    self.in_path_data_1 = in_path_data_1 # ./BSDS300/images
    self.in_path_data_2 = in_path_data_2 # ./BSDS300/images
    self.in_path_data_3 = in_path_data_3 # ./BSDS300/images
    self.img_size = img_size # (180, 180)

    self.imgs_data_1 = os.listdir(self.in_path_data_1)
    self.imgs_data_2 = os.listdir(self.in_path_data_2)
    self.imgs_data_3 = os.listdir(self.in_path_data_3)
    #self.sigma = sigma
    self.imgs_path = list()
    for i in self.imgs_data_1:
      _, ext = os.path.splitext(i)
      if ext in [".jpg", ".jpg", ".bmp", ".JPEG", ".jpeg"]:
        self.imgs_path.append(os.path.join(self.in_path_data_1, i))
    for i in self.imgs_data_2:
      _, ext = os.path.splitext(i)
      if ext in [".jpg", ".jpg", ".bmp", ".JPEG", ".jpeg"]:
        self.imgs_path.append(os.path.join(self.in_path_data_2, i))
    for i in self.imgs_data_3:
      _, ext = os.path.splitext(i)
      if ext in [".jpg", ".jpg", ".bmp", ".JPEG", ".jpeg"]:
        self.imgs_path.append(os.path.join(self.in_path_data_3, i))
    print(len(self.imgs_path))
    print(self.imgs_path)
  def __len__(self):
    #return len(self.imgs_path)
    return 8000*128
  
  def __getitem__(self, idx):
    #img_path = os.path.join(self.img_dir, self.imgs[idx])
    img_path = self.imgs_path[idx]
    clean_img = Image.open(img_path).convert('RGB')
    if (clean_img.size[0] > self.img_size[0]):
        left = np.random.randint(clean_img.size[0] - self.img_size[0])
    else:
        left = 0
    if (clean_img.size[1] > self.img_size[1]):
        top = np.random.randint(clean_img.size[1] - self.img_size[1])
    else:
        top = 0
    # .crop(left, upper, right, lower)
    cropped_clean = clean_img.crop([left, top, left+self.img_size[0], top+self.img_size[1]])
    transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ground_truth = transform(cropped_clean)
    noisy = addNoise(ground_truth, device)
    return noisy, ground_truth


class NoisyDatasetUnsup(torch.utils.data.Dataset):
  def __init__(self, in_path, in_path_coco, mode='train', img_size=(320, 320), sigma=30):
    super(NoisyDatasetUnsup, self).__init__()

    self.mode = mode #train or test
    self.in_path = in_path #
    self.img_size = img_size # (180, 180)

    self.img_dir = os.path.join(in_path, "Data")
    self.imgs = os.listdir(self.img_dir)
    i = 0
    for n in self.imgs:
      self.imgs[i] = os.path.join(n, "GT_SRGB_010.PNG")
      i = i+1
    #self.sigma = sigma

    self.imgs_path = list()
    for i in self.imgs:
      self.imgs_path.append(os.path.join(self.img_dir, i))

    self.in_path_coco = in_path_coco # ./unlabeled2017
    self.imgs_path += [self.in_path_coco + "/" + f for f in os.listdir(self.in_path_coco ) if os.path.isfile(os.path.join(self.in_path_coco , f))]

  def __len__(self):
    return len(self.imgs_path)

  def __getitem__(self, idx):
    #img_path = os.path.join(self.img_dir, self.imgs[idx])
    img_path = self.imgs_path[idx]
    clean_img = Image.open(img_path).convert('RGB')

    if (clean_img.size[0] > self.img_size[0]):
        left = np.random.randint(clean_img.size[0] - self.img_size[0])
    else:
        left = 0
    if (clean_img.size[1] > self.img_size[1]):
        top = np.random.randint(clean_img.size[1] - self.img_size[1])
    else:
        top = 0
    # .crop(left, upper, right, lower)
    cropped_clean = clean_img.crop([left, top, left+self.img_size[0], top+self.img_size[1]])
    transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ground_truth = transform(cropped_clean)
    noisy = addNoise(ground_truth, device)
    return noisy, ground_truth