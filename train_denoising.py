# -*- coding: utf-8 -*-
from pytorch_tcr import TCR
import torch
import torch.optim as optim
import torch.nn as nn
from parsing import get_parser
from models.FFDNet import FFDNet
from dataloaders import DataLoaderDenoising
from tqdm import tqdm
from time import sleep

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args = get_parser().parse_args()

denoise_model = FFDNet(3,3,96,15).to(device)
optimizer = torch.optim.Adam(denoise_model.parameters(), lr =args.lr)
criterion_mse = nn.MSELoss().to(device)
criterion_l1Loss = nn.L1Loss().to(device)

dataLoaderDenoising = DataLoaderDenoising(args.batch_size, args.workers)
trainloader = dataLoaderDenoising.get_trainloader()
trainloader_un = dataLoaderDenoising.get_trainloader_un()
validationloader = dataLoaderDenoising.get_validationloader()

tcr = TCR().to(device)
for epoch in range(args.epochs):   
    running_loss = 0.0
    for iteration, (data_sup, data_un) in enumerate(tqdm(zip(trainloader, trainloader_un), total=len(trainloader)), 0):
        #data_sup, data_un = batch[0] , batch[1]
        input, target = data_sup[0].to(device), data_sup[1].to(device)   # Here the data is used in supervised fashion
        input_un, target_un = data_un[0].to(device), data_un[1].to(device)   # Here the labels are not used
        
        optimizer.zero_grad()
        outputs = denoise_model(input)
        loss = criterion_mse(outputs,target)
        
        if args.with_tcr > 0:
            bs=  input_un.shape[0]
            random=torch.rand((bs, 1))
            transformed_input= tcr(input_un,random.to(device))
            loss_tcr= criterion_mse(denoise_model(transformed_input), tcr(denoise_model(input_un),random))
            total_loss= loss + args.weight_tcr*loss_tcr
            print("Loss TCR %f", loss_tcr)
        else:
            total_loss= loss

        running_loss += total_loss.item()
        total_loss.backward()
        optimizer.step()
        sleep(0.01)
        torch.save(denoise_model.state_dict(), "model_checkpoint_" + str(epoch+1)+ ".pt")

    print('Epoch-{0} lr: {1}'.format(epoch+1, optimizer.param_groups[0]['lr']))
    print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss ))      

print('Finished Training')
