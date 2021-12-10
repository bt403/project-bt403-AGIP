from pytorch_tcr import TCR
import torch
import torch.optim as optim
import torch.nn as nn
from parsing import get_parser
from models.FFDNet import FFDNet
from dataloaders import DataLoaderDenoising
from tqdm import tqdm
from time import sleep
import wandb


wandb.init(project="my-test-project", entity="btafur")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
args = get_parser().parse_args()

wandb.config = {
  "learning_rate": args.lr,
  "epochs": args.epochs,
  "batch_size": args.batch_size
}

denoise_model = FFDNet(3,3,96,15).to(device)
tcr = TCR().to(device)
optimizer = torch.optim.Adam(denoise_model.parameters(), lr =args.lr)

if (args.resume > 0):
    checkpoint = torch.load(args.checkpoint)
    denoise_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

denoise_model.train()

criterion_mse = nn.MSELoss().to(device)
criterion_l1Loss = nn.L1Loss().to(device)

dataLoaderDenoising = DataLoaderDenoising(args.batch_size, args.batch_size_un, args.workers)
trainloader = dataLoaderDenoising.get_trainloader()
trainloader_un = dataLoaderDenoising.get_trainloader_un()

def train(data_sup, data_un, denoise_model, running_loss, with_tcr, step):
    b_size = data_sup[0].shape[0]
    input, target = data_sup[0].to('cuda:0', non_blocking=True), data_sup[1].to('cuda:0', non_blocking=True)  # Here the data is used in supervised fashion
    if (with_tcr):
        b_size_unsup = data_un[0].shape[0]
        input_un, target_un = data_un[0].to('cuda:0', non_blocking=True), data_un[1].to('cuda:0', non_blocking=True)  # Here the labels are not used
    optimizer.zero_grad()
    outputs = denoise_model(input, b_size=b_size)
    loss = criterion_mse(outputs,target)
    if ((step+1)%200==0):
        wandb.log({"train_loss": loss}) 
    if with_tcr:
        bs = input_un.shape[0]
        random = torch.rand((bs, 1))
        transformed_input = tcr(input_un,random.to('cuda:0', non_blocking=True))
        loss_tcr = criterion_mse(denoise_model(transformed_input, b_size=b_size_unsup), tcr(denoise_model(input_un, b_size=b_size_unsup),random))
        total_loss= loss + args.weight_tcr*loss_tcr
    else:
        total_loss= loss
        

    running_loss += total_loss.item()
    total_loss.backward()
    optimizer.step()
    if with_tcr:
        return (running_loss, loss_tcr)
    return running_loss

if args.with_tcr > 0:
    for epoch in range(args.epochs):   
        running_loss = 0.0
        total_iter = min(len(trainloader), len(trainloader_un))
        step_log = 0
        for iteration, (data_sup, data_un) in enumerate(tqdm(zip(trainloader, trainloader_un), total=total_iter)):
            running_loss, loss_tcr = train(data_sup, data_un, denoise_model, running_loss, args.with_tcr, step_log)
            step_log += 1
        if ((epoch+1)%10 == 0):
            torch.save(denoise_model.state_dict(), "./checkpoint/denoise_checkpoint_with_tcr_" + str(epoch+1)+ ".pt")
        print('Epoch-{0} lr: {1}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        print('[%d] total loss: %.3f' % (epoch + 1, running_loss ))     
        print('tcr loss: %.3f' % (loss_tcr))  
        #wandb.log({"train_loss": running_loss})   
else:
    for epoch in range(args.epochs):   
        running_loss = 0.0
        total_iter = len(trainloader)
        step_log = 0
        for iteration, data_sup in enumerate(tqdm(trainloader, total=total_iter)):
            running_loss = train(data_sup, None, denoise_model, running_loss, args.with_tcr, step_log)
            step_log += 1
        if ((epoch+1)%2 == 0):
            torch.save(denoise_model.state_dict(), "./checkpoint/denoise_checkpoint_" + str(epoch+1)+ ".pt")
            torch.save({
            'epoch': epoch,
            'model_state_dict': denoise_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            }, "./checkpoint/denoise_checkpoint_resume_" + str(epoch+1)+ ".tar")
        print('Epoch-{0} lr: {1}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        print('[%d] loss: %.3f' % (epoch + 1, running_loss ))   
        #wandb.log({"train_loss": running_loss})   

print('Finished Training')
