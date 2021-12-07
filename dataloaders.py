import torch
from datasets.dataset_denoising import NoisyDataset, NoisyDatasetUnsup

dataset_dir = "./BSDS300/images"
dataset_dir_iapdr = "./iaprtc12/images"
dataset_dir_un = "./SIDD_Small_sRGB_Only"

class DataLoaderDenoising():
    def __init__(self, batch_size, workers):
        super(DataLoaderDenoising, self).__init__()
        self.trainloader = torch.utils.data.DataLoader(NoisyDataset(dataset_dir, dataset_dir_iapdr), batch_size=batch_size, shuffle=True, num_workers=workers)
        self.trainloader_un = torch.utils.data.DataLoader(NoisyDatasetUnsup(dataset_dir_un), batch_size=batch_size, shuffle=True, num_workers=workers) # The batch size for unsupervised data is more than supervised data
        self.validationloader = torch.utils.data.DataLoader(NoisyDataset(dataset_dir, dataset_dir_iapdr, mode='test', img_size=(320, 320)), batch_size=batch_size, shuffle=True, num_workers=workers)
        
    def get_trainloader(self):
        return self.trainloader
    
    def get_trainloader_un(self):
        return self.trainloader_un
    
    def get_validationloader(self):
        return self.validationloader