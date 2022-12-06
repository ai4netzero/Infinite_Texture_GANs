''' train
   This script train a generative model '''
import argparse
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm

from utils import *

#torch.set_num_threads(16) 
#import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

# configurations
parser = prepare_parser()

args = parser.parse_args()

# Device
device = prepare_device(args)

#Seeds
seed  = prepare_seed(args)
   
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
    

#parameters
batch_size = args.batch_size
G_b_size = args.G_batch_size
if G_b_size is None:
    G_b_size = batch_size

disc_iters = args.disc_iters
loss_fun = args.loss
epochs = args.epochs
cgan =  args.cgan
zdim = args.zdim
img_ch = args.img_ch
saving_rate = args.save_rate

    
#hyperparameres
lr_D = args.lr_D
lr_G = args.lr_G
beta1 = args.beta1
beta2 = args.beta2


dataloader,train_data = prepare_data(args)

print('Training samples: ',len(train_data))

# conditional GAN
#if cgan:
n_cl = args.n_cl
#else:
#    n_cl = 0


#models
netG,netD = prepare_models(args,n_cl,device)

if args.ema:
    netG_ema,_ = prepare_models(args,n_cl,device)
    with torch.no_grad():
        for key in netG_ema.state_dict():
            netG_ema.state_dict()[key].data.copy_(netG.state_dict()[key].data)
            
    for p in netG_ema.parameters():
        p.requires_grad = False
        


# OPTIMZERS 
optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, beta2))

# use decaying learning ratexs
if args.decay_lr == 'exp':
    schedulerD = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)
elif args.decay_lr == 'step':
    MILESTONES = [40,80,120] #None
    SCHEDULER_GAMMA = 0.5
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=MILESTONES, 
                                               gamma=SCHEDULER_GAMMA, last_epoch=-1)
    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=MILESTONES, 
                                                gamma=SCHEDULER_GAMMA, last_epoch=-1)   
    
#saved_models    
if args.saved_cp is not None:
    netG,netD,optimizerG,optimizerD,st_epoch,G_losses,D_losses = load_from_saved(args,netG,netD,optimizerG,optimizerD)
else:
    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    st_epoch = 1


# Parallel GPU if ngpu > 1
if (device.type == 'cuda') and (args.ngpu > 1):
    netG = nn.DataParallel(netG, args.gpu_list)
    netD = nn.DataParallel(netD, args.gpu_list)
    netG_ema = nn.DataParallel(netG_ema,args.gpu_list)

#print(list(range(args.dev_num,ngpu)))

#settings for losses
if loss_fun == 'standard':
    dis_criterion = nn.BCEWithLogitsLoss().to(device)
    #labels
    if args.smooth:
        label_t = 0.9
        label_f = 0
    else:
        label_t = 1
        label_f = 0
        

filename = prepare_filename(args)
                   

# Print the model
print(netG)
print(netD)
print("# Params. G: ", sum(p.numel() for p in netG.parameters()))
print("# Params. D: ", sum(p.numel() for p in netD.parameters()))


    
TIME_LIMIT = args.limit
start_time = time.time()


if args.use_coord:
    args.img_res = (2**netG.n_layers_G) * args.base_res
    meta_coord_grid = create_coord_gird(args.meta_grid_h* args.img_res,args.meta_grid_w* args.img_res,coef = args.period_coef)
else:
    meta_coord_grid = None      



def train(num_epochs=1, disc_iters=1):
    global G_losses, D_losses
    
    print("Starting Training Loop...")
    # For each epoch
    
    for epoch in range(st_epoch,num_epochs+1):
        if TIME_LIMIT is not None and elapsed_time(start_time) > TIME_LIMIT:
            print('Time limit reached')
            break
        D_running_loss = 0
        G_running_loss = 0
        running_examples_D = 0
        running_examples_G = 0


        #print(epoch)
        # For each mini-batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            #print(i)
            real_x = data[0].to(device)
            # load labels if using cgan 
            if args.cgan:
                real_y = data[1].float().to(device) # discrete 0,1,...n_cl-1
                if args.ohe: # convert discrete values to ohe
                    real_y = disc_2_ohe(real_y.long(),n_cl,device)
                elif args.real_cond_list is not None:
                    real_y = disc_2_cont(real_y,args.real_cond_list,device)
            else :
                real_y = None

            b_size = real_x.size(0)
            #G_b_size = b_size
            # Update D network
            for _ in range(disc_iters): 
                netD.zero_grad()
                # update with real labels
                real_logit = netD(real_x, real_y)
                if loss_fun == 'hinge':
                    D_loss_real = torch.mean(F.relu(1.0 - real_logit))
                elif loss_fun == 'standard':
                    adv_labels = torch.FloatTensor(1).fill_(label_t).expand_as(real_logit).to(device)
                    D_loss_real = dis_criterion(real_logit,adv_labels)
                elif loss_fun == 'wgan':
                    D_loss_real = -torch.mean(real_logit)
                    
                D_loss_real.backward()

                # update with fake labels
                if args.G_patch_1D:
                    fake_x, fake_y = sample_patches_from_gen_1D(args,G_b_size, zdim,args.zdim_b,args.num_patches_per_img, n_cl, netG,device,real_y=real_y)
                    fake_x = merge_patches_1D(fake_x,args.num_patches_per_img,device)
                elif args.G_patch_2D:
                    fake_x, fake_y = sample_patches_from_gen_2D(args,G_b_size, netG,meta_coord_grid,device)
                    fake_x = merge_patches_2D(fake_x,h = args.num_patches_h,w = args.num_patches_w,device = device)
                else:
                    fake_x, fake_y = sample_from_gen(args,G_b_size, zdim, n_cl, netG,device,real_y=real_y)
                fake_logit = netD(fake_x.detach(),fake_y)
                    
                if loss_fun == 'hinge':  
                    D_loss_fake = torch.mean(F.relu(1.0 + fake_logit))
                elif loss_fun == 'standard':
                    adv_labels = torch.FloatTensor(1).fill_(label_f).expand_as(fake_logit).to(device)
                    D_loss_fake = dis_criterion(fake_logit,adv_labels) 
                elif loss_fun == 'wgan':
                    D_loss_fake = torch.mean(fake_logit)


                D_loss_fake.backward()
                optimizerD.step()
                D_running_loss += (D_loss_fake.item()*G_b_size + D_loss_real.item()*b_size)
                
           # Update G
            netG.zero_grad()
            if args.x_fake_GD is False:
                if args.G_patch_1D:
                    fake_x, fake_y = sample_patches_from_gen_1D(args,G_b_size, zdim,args.zdim_b,args.num_patches_per_img, n_cl, netG,device,real_y=real_y)
                    fake_x = merge_patches_1D(fake_x,args.num_patches_per_img)
                elif args.G_patch_2D:
                    fake_x, fake_y = sample_patches_from_gen_2D(args,G_b_size, netG,meta_coord_grid,device)
                    fake_x = merge_patches_2D(fake_x,h = args.num_patches_h,w = args.num_patches_w,device = device)
                else:
                    fake_x, fake_y = sample_from_gen(args,G_b_size, zdim, n_cl, netG,device,real_y=real_y)
            fake_logit = netD(fake_x,fake_y)

            if loss_fun == 'hinge':
                _G_loss = -torch.mean(fake_logit)
            elif loss_fun == 'standard':
                adv_labels = torch.FloatTensor(1).fill_(label_t).expand_as(fake_logit).to(device)
                _G_loss = dis_criterion(fake_logit, adv_labels)
            elif loss_fun == 'wgan':
                _G_loss = -torch.mean(fake_logit)
                
            _G_loss.backward()
            optimizerG.step()
            G_running_loss += _G_loss.item()*G_b_size
            
            running_examples_D+= b_size
            running_examples_G+= G_b_size
            
            if args.ema:
                with torch.no_grad():
                    for key in netG.state_dict():
                        netG_ema.state_dict()[key].data.copy_(netG_ema.state_dict()[key].data * args.ema_decay 
                                                 + netG.state_dict()[key].data * (1 - args.ema_decay))
            
        
        if args.decay_lr:
            schedulerD.step()
            schedulerG.step()
         
        D_running_loss/=running_examples_D
        G_running_loss/=running_examples_G
        
        # Output training stats
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f, elapsed_time = %.4f min'
              % (epoch, num_epochs,
                 D_running_loss, G_running_loss,elapsed_time(start_time)/60))
                    
        # Save Losses for plotting later
        G_losses.append(G_running_loss)
        D_losses.append(D_running_loss)

        
        if saving_rate is not None and (epoch%saving_rate ==0 or epoch == epochs)  :
            # saving and showing results
            torch.save({
                        'epoch': epoch,
                        'netG_state_dict': netG.state_dict(),
                        'netD_state_dict': netD.state_dict(),
                        'optimizerG_state_dict': optimizerG.state_dict(),
                        'optimizerD_state_dict': optimizerD.state_dict(),
                        'Gloss':  G_losses,
                        'Dloss':  D_losses,
                        'args': args,
                        'seed': seed,
                        }, filename+str(epoch) +".pth")
            
train(epochs,disc_iters)

if args.ema:
    torch.save({
                'netG_state_dict': netG_ema.state_dict(),
                'args': args,
                }, filename+"_ema.pth")

        
fig1 = plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
fig1.savefig(filename + 'losses.png')

torch.cuda.empty_cache()
