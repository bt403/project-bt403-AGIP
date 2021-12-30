from pytorch_tcr import TCR
import torch
import torch.optim as optim
import torch.nn as nn
from parsing import get_parser
from models.FFDNet import FFDNet
from dataloaders import DataLoaderDenoising
from tqdm import tqdm
import math
import numpy as np
from torch.autograd import Variable
import wandb
from skimage.metrics import peak_signal_noise_ratio

wandb.init(project="my-test-project", entity="btafur")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
args = get_parser().parse_args()

wandb.config = {
  "learning_rate": args.lr,
  "epochs": args.epochs,
  "batch_size": args.batch_size
}
wandb.init()

def weights_init_kaiming(lyr):
	r"""Initializes weights of the model according to the "He" initialization
	method described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution.
	This function is to be called by the torch.nn.Module.apply() method,
	which applies weights_init_kaiming() to every layer of the model.
	"""
	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm') != -1:
		lyr.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).\
			clamp_(-0.025, 0.025)
		nn.init.constant(lyr.bias.data, 0.0)

denoise_model = FFDNet(3).to(device)
denoise_model.apply(weights_init_kaiming)
tcr = TCR().to(device)



device_ids = [0]
denoise_model_p = nn.DataParallel(denoise_model, device_ids=device_ids).cuda()

criterion_mse = nn.MSELoss(size_average=False).to(device)
criterion_l1Loss = nn.L1Loss().to(device)
optimizer = torch.optim.Adam(denoise_model_p.parameters(), lr =args.lr)

if (args.resume > 0):
    checkpoint = torch.load(args.checkpoint)
    denoise_model_p.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']



dataLoaderDenoising = DataLoaderDenoising(args.batch_size, args.batch_size_un, args.workers)
trainloader = dataLoaderDenoising.get_trainloader()
trainloader_un = dataLoaderDenoising.get_trainloader_un()
validationloader = dataLoaderDenoising.get_validationloader()
val_noiseL = 50.0
noiseIntL = [0, 75]
val_noiseL /= 255.
noiseIntL[0] /= 255.
noiseIntL[1] /= 255.

def batch_psnr(img, imclean, data_range):
	r"""
	Computes the PSNR along the batch dimension (not pixel-wise)

	Args:
		img: a `torch.Tensor` containing the restored image
		imclean: a `torch.Tensor` containing the reference image
		data_range: The data range of the input image (distance between
			minimum and maximum possible values). By default, this is estimated
			from the image data-type.
	"""
	img_cpu = img.data.cpu().numpy().astype(np.float32)
	imgclean = imclean.data.cpu().numpy().astype(np.float32)
	psnr = 0
	for i in range(img_cpu.shape[0]):
		psnr += peak_signal_noise_ratio(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
					   data_range=data_range)
	return psnr/img_cpu.shape[0]

def validate():
    avg_psnr = 0
    with torch.no_grad():
        for batch in validationloader:
            input, target = batch[0].to(device), batch[1].to(device)
            img_val = input
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=val_noiseL)
            imgn_val = img_val + noise.to(device)
            img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
            sigma_noise = Variable(torch.cuda.FloatTensor([val_noiseL]))
            out_val = torch.clamp(imgn_val-denoise_model_p(imgn_val, sigma_noise), 0., 1.)
            avg_psnr += batch_psnr(out_val, img_val, 1.)
    wandb.log({"psnr": avg_psnr / len(validationloader)}) 
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(validationloader)))

def train(data_sup, data_un, denoise_model_p, running_loss, with_tcr, step):
    denoise_model_p.train()
    denoise_model_p.zero_grad()
    optimizer.zero_grad()

    img_train = data_sup.to('cuda:0', non_blocking=True) # Here the data is used in supervised fashion
    if (with_tcr):
        img_un = data_un.to('cuda:0', non_blocking=True)

    noise = torch.zeros(img_train.size())
    stdn = np.random.uniform(noiseIntL[0], noiseIntL[1], \
                    size=noise.size()[0])
    for nx in range(noise.size()[0]):
        sizen = noise[0, :, :, :].size()
        noise[nx, :, :, :] = torch.FloatTensor(sizen).\
                            normal_(mean=0, std=stdn[nx])
    imgn_train = img_train + noise.to(device)
    # Create input Variables
    img_train = Variable(img_train.cuda())
    imgn_train = Variable(imgn_train.cuda())
    noise = Variable(noise.cuda())
    stdn_var = Variable(torch.cuda.FloatTensor(stdn))
    #print(imgn_train.shape)
    # Evaluate model and optimize it
    out_train = denoise_model_p(imgn_train, stdn_var)
    loss = criterion_mse(out_train, noise) / (imgn_train.size()[0]*2)

    if with_tcr:
        bs = img_un.shape[0]
        random = torch.rand((bs, 1))
        noise = torch.zeros(img_un.size())
        stdn = np.random.uniform(noiseIntL[0], noiseIntL[1], \
                        size=noise.size()[0])
        for nx in range(noise.size()[0]):
            sizen = noise[0, :, :, :].size()
            noise[nx, :, :, :] = torch.FloatTensor(sizen).\
                                normal_(mean=0, std=stdn[nx])
        imgn_un = img_un + noise.to(device)
        # Create input Variables
        img_un = Variable(img_un.cuda())
        noise = Variable(noise.cuda())
        imgn_un = Variable(imgn_un.type(torch.FloatTensor).cuda())
        stdn_var_un = Variable(torch.cuda.FloatTensor(stdn))
        transformed_input = tcr(imgn_un,random.to('cuda:0', non_blocking=True))
        loss_tcr = criterion_mse(denoise_model_p(transformed_input, stdn_var_un), tcr(denoise_model_p(imgn_un, stdn_var_un),random)) / (imgn_un.size()[0]*2)
        total_loss= loss 
        #+ args.weight_tcr*loss_tcr
    else:
        total_loss= loss

    total_loss.backward()
    optimizer.step()
    running_loss += total_loss
    if (step < 50):
        print("total loss: ", str(total_loss))      
        print("loss: ", str(loss))      
        if with_tcr:
            print("tcr loss: ", str(loss_tcr))   
    if ((step+1)%500==0):
        denoise_model_p.eval()
        out_train = torch.clamp(imgn_train-denoise_model_p(imgn_train, stdn_var), 0., 1.)
        psnr_train = batch_psnr(out_train, img_train, 1.)
        print("PSNR Train: " + str(psnr_train))
        print("total loss: ", str(total_loss))      
        print("loss: ", str(loss))      
        if with_tcr:
            print("tcr loss: ", str(loss_tcr))      
        validate()
        wandb.log({"train_loss": total_loss}) 

    if with_tcr:
        return (running_loss, loss_tcr)

    return running_loss

if args.with_tcr > 0:
    for epoch in range(args.epochs):   
        running_loss = 0.0
        total_iter = min(len(trainloader), len(trainloader_un))
        step_log = 0
        for iteration, (data_sup, data_un) in enumerate(tqdm(zip(trainloader, trainloader_un), total=total_iter)):
            running_loss, loss_tcr = train(data_sup, data_un, denoise_model_p, running_loss, args.with_tcr, step_log)
            step_log += 1
        if ((epoch+1)%2 == 0):
            torch.save(denoise_model_p.state_dict(), "./checkpoint/denoise_checkpoint_with_tcr_" + str(epoch+1)+ ".pt")
            torch.save({
            'epoch': epoch,
            'model_state_dict': denoise_model_p.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            }, "./checkpoint/denoise_checkpoint_with_tcr_resume_" + str(epoch+1)+ ".tar")
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
            running_loss = train(data_sup, None, denoise_model_p, running_loss, args.with_tcr, step_log)
            step_log += 1
        if ((epoch+1)%2 == 0):
            torch.save(denoise_model_p.state_dict(), "./checkpoint/denoise_checkpoint_" + str(epoch+1)+ ".pt")
            torch.save({
            'epoch': epoch,
            'model_state_dict': denoise_model_p.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            }, "./checkpoint/denoise_checkpoint_resume_" + str(epoch+1)+ ".tar")
        print('Epoch-{0} lr: {1}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        print('[%d] loss: %.3f' % (epoch + 1, running_loss ))   
        #wandb.log({"train_loss": running_loss})   

print('Finished Training')
