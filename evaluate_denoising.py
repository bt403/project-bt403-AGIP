# -*- coding: utf-8 -*-
import pytorch_tcr as TCR
import torch
import torch.optim as optim
import torch.nn as nn
from parsing import get_parser
from models.FFDNet import FFDNet
from dataloaders import DataLoaderDenoising
from torch.autograd import Variable
import numpy as np
import math

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args = get_parser().parse_args()

denoise_model = FFDNet(3)
device_ids = [0]
denoise_model_p = nn.DataParallel(denoise_model, device_ids=device_ids).cuda()

#denoise_model_p.load_state_dict(torch.load(args.checkpoint))
checkpoint = torch.load(args.checkpoint)
denoise_model_p.load_state_dict(checkpoint['model_state_dict'])
denoise_model_p.to(device)
denoise_model_p.eval()

dataLoaderDenoising = DataLoaderDenoising(args.batch_size, args.batch_size_un, args.workers)

validationloader = dataLoaderDenoising.get_validationloader()
if (args.val_un > 0):
    validationloader = dataLoaderDenoising.get_validationloader_un()
    
criterion_mse = nn.MSELoss().to(device)
val_noiseL = args.noise_level
val_noiseL /= 255.
sigma_noise = Variable(torch.cuda.FloatTensor([val_noiseL]))

def validate():
    avg_psnr = 0
    print(len(validationloader))
    with torch.no_grad():
        for batch in validationloader:
            input, target = batch[0].to(device), batch[1].to(device)
            noise = torch.FloatTensor(input.size()).normal_(mean=0, std=val_noiseL)
            imgn_val = input + noise.to(device)
            img_val, imgn_val = Variable(input.cuda()), Variable(imgn_val.cuda())
            sigma_noise = Variable(torch.cuda.FloatTensor([val_noiseL]))
            out_val = torch.clamp(imgn_val-denoise_model_p(imgn_val, sigma_noise), 0., 1.)

            mse = criterion_mse(out_val, target)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(validationloader)))

validate()