import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim

from utils import prepare_data,prepare_device,prepare_filename,prepare_models,prepare_parser,prepare_seed,elapsed_time, sample_from_gen_PatchByPatch_train,sample_from_gen

def train(args):
    

    # Device
    device = prepare_device(args)

    #Seeds
    seed  = prepare_seed(args)
    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    print(args)
    
    # Training loaders
    dataloader,train_data = prepare_data(args)

    print('Training samples: ',len(train_data))
 

    #models
    netG,netD = prepare_models(args,device)

    # Building a generator network that applied exponential moving average to the weights over the epochs.
    if args.ema:
        netG_ema,_ = prepare_models(args,device)
        with torch.no_grad():
            for key in netG_ema.state_dict():
                netG_ema.state_dict()[key].data.copy_(netG.state_dict()[key].data)
                
        for p in netG_ema.parameters():
            p.requires_grad = False


    # Print the model
    print(netG)
    print(netD)
    print("# Params. G: ", sum(p.numel() for p in netG.parameters()))
    print("# Params. D: ", sum(p.numel() for p in netD.parameters()))



    # OPTIMZERS 
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_D, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_G, betas=(args.beta1, args.beta2))

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


    # Parallel GPUs if ngpu > 1
    if (device.type == 'cuda') and (args.num_gpus > 1):
        netG = nn.DataParallel(netG, args.gpu_list)
        netD =  nn.DataParallel(netD, args.gpu_list)
        netG_ema =  nn.DataParallel(netG_ema,args.gpu_list)


    #settings for losses
    dis_criterion = nn.BCEWithLogitsLoss().to(device)

    # discriminator labels (with smoothing option)
    if args.smooth:
        label_t = 0.9
        label_f = 0
    else:
        label_t = 1
        label_f = 0
            

    filename = prepare_filename(args)
                    


    start_time = time.time()

    # Lists to store the G and D losses
    G_losses = []
    D_losses = []
        
    
    print("Starting Training Loop...")
    # For each epoch
    
    for epoch in range(args.epochs):
        D_running_loss = 0
        G_running_loss = 0
        running_examples_D = 0
        running_examples_G = 0

        # For each mini-batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            # Fetch the real data
            real_x = data[0].to(device)
            b_size = real_x.size(0)
            
            #save_image(real_x[0]*0.5+0.5,'real.jpg')
            #exit()
            
            # Update D network
            # # max⁡〖𝔼_𝑦 [log⁡〖𝐷(𝑦)〗] +  𝔼_(𝑧,𝑀) [log⁡〖(1−𝐷(𝑋)〗)]〗   
            for _ in range(args.disc_iters): 
                
                netD.zero_grad()

                # Update with real data, Term 1: max⁡〖𝔼_𝑦 [log⁡〖𝐷(𝑦)〗] 
                real_logit = netD(real_x) # 𝐷(y)
                
                adv_labels = torch.FloatTensor(1).fill_(label_t).expand_as(real_logit).to(device) # ones
                D_loss_real = dis_criterion(real_logit,adv_labels) #  l  = - [y log[y_hat] + (1-y) log[(1-y_hat)]] ==> -log⁡〖𝐷(𝑦)〗

                D_loss_real.backward()
                
                
                # Update with fake data, Term 2: max 𝔼_(𝑧,𝑀) [log⁡〖(1−𝐷(𝑋)〗)]
                
                if args.padding_mode == 'local':
                    fake_x = sample_from_gen_PatchByPatch_train(netG,args.z_dim,args.base_res,args.map_dim,num_images=args.num_images,
                                                      num_patches_height=args.num_patches_height,num_patches_width=args.num_patches_width,device=device)
                else:
                    fake_x =  sample_from_gen(netG,args.z_dim,args.base_res,num_images=args.num_images,device =device) 

                    
                fake_logit = netD(fake_x.detach()) # 𝐷(𝑋)
                    
                adv_labels = torch.FloatTensor(1).fill_(label_f).expand_as(fake_logit).to(device) # zeros
                D_loss_fake = dis_criterion(fake_logit,adv_labels)  #  l  = - [y log[y_hat] + (1-y) log[(1-y_hat)]] ==> -log⁡〖(1−𝐷(𝑋)〗)


                D_loss_fake.backward()
                optimizerD.step()
                D_running_loss += (D_loss_fake.item()*args.num_images + D_loss_real.item()*b_size)
            
            
            # Update G
            # min⁡〖𝔼_𝑦 [log⁡〖𝐷(𝑦)〗] +  𝔼_(𝑧,𝑀) [log⁡〖(1−𝐷(𝑋)〗)]〗
            # min  𝔼_(𝑧,𝑀) [log⁡〖(1−𝐷(𝑋)〗)]
            # max  𝔼_(𝑧,𝑀) [log⁡〖(𝐷(𝑋)〗)]
            netG.zero_grad()
            fake_logit = netD(fake_x) # 𝐷(𝑋)

            adv_labels = torch.FloatTensor(1).fill_(label_t).expand_as(fake_logit).to(device) # ones
            _G_loss = dis_criterion(fake_logit, adv_labels) #  l  = - [y log[y_hat] + (1-y) log[(1-y_hat)]] ==> - log⁡〖(𝐷(𝑋)〗)

                
            _G_loss.backward()
            optimizerG.step()
            
            G_running_loss += _G_loss.item()*args.num_images
            
            running_examples_D+= b_size
            running_examples_G+= args.num_images
            
            if args.ema:
                with torch.no_grad():
                    for key in netG.state_dict():
                        netG_ema.state_dict()[key].data.copy_(netG_ema.state_dict()[key].data * args.ema_decay 
                                                 + netG.state_dict()[key].data * (1 - args.ema_decay))
            
        # apply learning rate schedulers 
        if args.decay_lr:
            schedulerD.step()
            schedulerG.step()
         
        D_running_loss/=running_examples_D
        G_running_loss/=running_examples_G
        
        # Printing training stats
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f, elapsed_time = %.4f min'
              % (epoch+1, args.epochs,
                 D_running_loss, G_running_loss,elapsed_time(start_time)/60))
                    
        # Save Losses for plotting
        G_losses.append(G_running_loss)
        D_losses.append(D_running_loss)

        # saving model checkpoints
        if args.saving_rate is not None and ((epoch+1)%args.saving_rate ==0 or (epoch+1) == args.epochs)  :
            torch.save({
                        'epoch': epoch+1,
                        'netG_state_dict': netG.state_dict(),
                        'netD_state_dict': netD.state_dict(),
                        'Gloss':  G_losses,
                        'Dloss':  D_losses,
                        'args': args,
                        'seed': seed,
                        }, filename+str(epoch+1) +".pth")
            
        # if last epoch then save the ema model and plot the losses
        if epoch+1 == args.epochs:
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


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()          
    train(args)
    
