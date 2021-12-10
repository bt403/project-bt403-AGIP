import torch
from datasets.dataset_denoising import NoisyDataset, NoisyDatasetUnsup, NoisyDatasetUn

dataset_dir = "./BSDS300/images"
dataset_dir_iapdr = "./iaprtc12/images"
dataset_dir_iapdr_val = "./iaprtc12/additional_images"
dataset_dir_un = "./SIDD_Small_sRGB_Only"
dataset_dir_un_coco = "./unlabeled2017"

dataset_dir_sup_1 = "./BBSDS500/train"
dataset_dir_sup_2 = "./exploration_database"
dataset_dir_sup_3 = "./imagenet-400"
dataset_dir_un_1_gt = "./CBSD68/original_png"
dataset_dir_un_1_noisy = "./CBSD68/noisy50"

class DataLoaderDenoising():
    def __init__(self, batch_size, batch_size_un, workers):
        super(DataLoaderDenoising, self).__init__()
        self.trainloader = torch.utils.data.DataLoader(NoisyDataset(dataset_dir_sup_1, dataset_dir_sup_2, dataset_dir_sup_3, 'train', (50,50), 30), batch_size=batch_size, shuffle=True, num_workers=workers)
        self.trainloader_un = torch.utils.data.DataLoader(NoisyDatasetUnsup(dataset_dir_un, dataset_dir_un_coco), batch_size=batch_size_un, shuffle=True, num_workers=workers) # The batch size for unsupervised data is more than supervised data
        self.validationloader = torch.utils.data.DataLoader(NoisyDatasetUn(dataset_dir_un_1_gt, dataset_dir_un_1_noisy, mode='test', img_size=(320, 320)), batch_size=batch_size, shuffle=True, num_workers=workers)
        
    def get_trainloader(self):
        return self.trainloader
    
    def get_trainloader_un(self):
        return self.trainloader_un
    
    def get_validationloader(self):
        return self.validationloader