import torch
import os
from PIL import Image
import numpy as np
import torchvision as tv
import random as rand
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from sklearn.model_selection import train_test_split

def get_imgs_path_train(in_path_data_1, in_path_data_2, in_path_data_3):
  imgs_data_1 = os.listdir(in_path_data_1)
  imgs_data_2 = os.listdir(in_path_data_2)
  imgs_data_3 = os.listdir(in_path_data_3)

  imgs_path = list()
  for i in imgs_data_1:
    _, ext = os.path.splitext(i)
    if ext in [".jpg", ".jpg", ".bmp", ".JPEG", ".jpeg"]:
      imgs_path.append(os.path.join(in_path_data_1, i))
  for i in imgs_data_2:
    _, ext = os.path.splitext(i)
    if ext in [".jpg", ".jpg", ".bmp", ".JPEG", ".jpeg"]:
      imgs_path.append(os.path.join(in_path_data_2, i))
  for i in imgs_data_3:
    _, ext = os.path.splitext(i)
    if ext in [".jpg", ".jpg", ".bmp", ".JPEG", ".jpeg"]:
      imgs_path.append(os.path.join(in_path_data_3, i))
  return imgs_path

def addNoise(x, sigma=None): 
    """
    We will use this helper function to add noise to some data. 
    x: the data we want to add noise to
    device: the CPU or GPU that the input is located on. 
    """
    #noiseLevel = rand.choice([0.5])
    if sigma is not None:
      noiseLevel = sigma
    else:
      noiseLevel = rand.choice([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    #return x + normal.sample(sample_shape=torch.Size(x.shape)).to(device)
    return x + (torch.randn(x.shape) * noiseLevel)

class NoisyDataset(torch.utils.data.Dataset):
  def __init__(self, in_path_data_1, in_path_data_2, in_path_data_3, mode='train', img_size=(320, 320), batch_size=128):
    super(NoisyDataset, self).__init__()

    self.mode = mode #train or test
    self.in_path_data_1 = in_path_data_1
    self.in_path_data_2 = in_path_data_2
    self.in_path_data_3 = in_path_data_3
    self.img_size = img_size # (180, 180)
    self.batch_size = batch_size
    
    self.imgs_path = get_imgs_path_train(self.in_path_data_1, self.in_path_data_2, self.in_path_data_3)

  def __len__(self):
    return 8000*self.batch_size
  
  def __getitem__(self, idx):
    idx = idx%len(self.imgs_path)
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
    cropped_clean = clean_img.crop([left, top, left+self.img_size[0], top+self.img_size[1]])
   
    transform = tv.transforms.Compose([
                                    tv.transforms.RandomHorizontalFlip(p=0.5),
                                    ])
    ground_truth = transform(cropped_clean)
    ground_truth = np.array(cropped_clean) / 255.
    ground_truth = tv.transforms.ToTensor()(ground_truth)
    #print(ground_truth[0])
    #noisy = addNoise(ground_truth)
    noisy = ground_truth
    return noisy, ground_truth


class NoisyDatasetVal(torch.utils.data.Dataset):
  def __init__(self, in_path_gt, in_path_noisy, mode='train', img_size=(320, 320), sigma=30):
    super(NoisyDatasetVal, self).__init__()

    self.mode = mode #train or test
    self.in_path_gt = in_path_gt #
    self.in_path_noisy = in_path_noisy #
    self.img_size = img_size # (180, 180)

    self.imgs_gt = os.listdir(self.in_path_gt)
    self.imgs_noisy = os.listdir(self.in_path_noisy)
    
    self.imgs_path_gt = list()
    for i in self.imgs_gt:
      _, ext = os.path.splitext(i)
      if ext in [".jpg", ".jpg", ".bmp", ".JPEG", ".jpeg", ".png"]:
        self.imgs_path_gt.append(os.path.join(self.in_path_gt, i))

    self.imgs_path_noisy = list()
    for i in self.imgs_noisy:
      _, ext = os.path.splitext(i)
      if ext in [".jpg", ".jpg", ".bmp", ".JPEG", ".jpeg", ".png"]:
        self.imgs_path_noisy.append(os.path.join(self.in_path_noisy, i))


  def __len__(self):
    return len(self.imgs_path_gt)

  def __getitem__(self, idx):
    #img_path = os.path.join(self.img_dir, self.imgs[idx])
    img_path_gt = self.imgs_path_gt[idx]
    img_path_noisy = self.imgs_path_noisy[idx]
    clean_img = Image.open(img_path_gt).convert('RGB')
    noisy_img = Image.open(img_path_noisy).convert('RGB')

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
    cropped_noisy = noisy_img.crop([left, top, left+self.img_size[0], top+self.img_size[1]])
    #transform = tv.transforms.Compose([tv.transforms.ToTensor(),])
    ground_truth = np.array(cropped_clean) / 255.
    ground_truth =  tv.transforms.ToTensor()(ground_truth)
    noisy = np.array(cropped_noisy) / 255.
    return ground_truth, ground_truth


class NoisyDatasetUn(torch.utils.data.Dataset):
  def __init__(self, in_path, mode='train', img_size=(320, 320), batch_size=1280):
    super(NoisyDatasetUn, self).__init__()
  
    self.batch_size = batch_size
    self.in_path = in_path #
    self.mode = mode #train or test
    self.imgs = os.listdir(self.in_path)
    self.imgs_path = list()
    self.img_size = img_size

    for i in self.imgs:
      _, ext = os.path.splitext(i)
      if ext in [".jpg", ".jpg", ".bmp", ".JPEG", ".jpeg", ".png"]:
        self.imgs_path.append(os.path.join(self.in_path, i))

    x_train ,x_test = train_test_split(self.imgs_path,test_size=0.3, random_state=42)
    if (self.mode == "val"):
      self.imgs_path = x_test
    else:
      self.imgs_path = x_train
      print("images_path")
      print(self.imgs_path)
      self.imgs_path_train = get_imgs_path_train(self.in_path_data_1, self.in_path_data_2, self.in_path_data_3)
      print("images_path train")
      print(self.imgs_path_train)
      self.imgs_path += self.imgs_path_train
      print("images_path total")
      print(self.imgs_path)

  def __len__(self):
    if (self.mode == "val"):
      return len(self.imgs_path)
    else:
      return 8000*self.batch_size

  def __getitem__(self, idx):
    idx = idx%len(self.imgs_path)
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
    cropped_clean = clean_img.crop([left, top, left+self.img_size[0], top+self.img_size[1]])
   
    transform = tv.transforms.Compose([
                                    tv.transforms.RandomHorizontalFlip(p=0.5),
                                    ])
    ground_truth = transform(cropped_clean)
    ground_truth = np.array(cropped_clean) / 255.
    ground_truth = tv.transforms.ToTensor()(ground_truth)
    #print(ground_truth[0])
    #noisy = addNoise(ground_truth)
    noisy = ground_truth
    return noisy, ground_truth