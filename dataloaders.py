import torch
from datasets.dataset_denoising import NoisyDataset, NoisyDatasetUn, NoisyDatasetVal

dataset_dir = "./BSDS300/images"
dataset_dir_iapdr = "./iaprtc12/images"
dataset_dir_iapdr_val = "./iaprtc12/additional_images"
dataset_dir_un = "./SIDD_Small_sRGB_Only"

dataset_dir_un_coco = "./unlabeled2017"
dataset_dir_sup_1 = "./BBSDS500/train"
dataset_dir_sup_2 = "./exploration_database"
dataset_dir_sup_3 = "./imagenet-400"
dataset_dir_val_1_gt = "./CBSD68/original_png/original_png"
dataset_dir_val_1_noisy = "./CBSD68/noisy50"


class DataLoaderDenoising():
    def __init__(self, batch_size, batch_size_un, workers):
        super(DataLoaderDenoising, self).__init__()
        self.trainloader = torch.utils.data.DataLoader(NoisyDataset(dataset_dir_sup_1, dataset_dir_sup_2, dataset_dir_sup_3, 'train', (50,50)), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        self.trainloader_un = torch.utils.data.DataLoader(NoisyDatasetUn(dataset_dir_un_coco, 'train', (50,50)), batch_size=batch_size_un, shuffle=True, num_workers=workers, pin_memory=True)
        self.validationloader_cbsd68 = torch.utils.data.DataLoader(NoisyDatasetVal(dataset_dir_val_1_gt, dataset_dir_val_1_noisy, mode='test', img_size=(480, 320)), batch_size=1, shuffle=True, num_workers=workers)
        self.validationloader_coco = torch.utils.data.DataLoader(NoisyDatasetUn(dataset_dir_un_coco, 'val', (50,50)), batch_size=batch_size_un, shuffle=True, num_workers=workers, pin_memory=True)
        
    def get_trainloader(self):
        return self.trainloader
    
    def get_trainloader_un(self):
        return self.trainloader_un
    
    def get_validationloader(self):
        return self.validationloader_cbsd68