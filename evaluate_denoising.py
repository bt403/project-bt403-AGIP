# -*- coding: utf-8 -*-
import pytorch_tcr as TCR
import torch
import torch.optim as optim
import torch.nn as nn
from parsing import get_parser
from models.FFDNet import FFDNet
from dataloaders import DataLoaderDenoising


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args = get_parser().parse_args()

denoise_model = FFDNet(3,3,96,15)
denoise_model.load_state_dict(torch.load(args.model_checkpoint))
denoise_model.to(device)
denoise_model.eval()

dataLoaderDenoising = DataLoaderDenoising(args.batch_size, args.workers)
validationloader = dataLoaderDenoising.get_validationloader()
criterion_mse = nn.MSELoss().to(device)

def validate():
    avg_psnr = 0
    with torch.no_grad():
        for batch in validationloader:
            input, target = batch[0].to(device), batch[1].to(device)
            prediction = denoise_model(input, b_size=len(input))
            mse = criterion_mse(prediction, target)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(validationloader)))

validate()