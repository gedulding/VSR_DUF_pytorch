import os
import math
import tqdm
import torch
import itertools
import traceback
import numpy as np
from tensorboardX import SummaryWriter
from torch import nn , optim
from net import DUFNET_16L
from utils import HuberLoss, load_checkpoint,load_checkpoint, save_checkpoint, ensure_dir,loadyaml
import yaml

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

def train(args, pt_dir, train_data_loader , train_label_loader):
    yamfile = './config/config.yaml'
    config = loadyaml(yamfile)
    lr = float(config['train']['adam']['lr'])
    step_size = int(config['train']['scheduler']['step_size'])
    gamma = float(config['train']['scheduler']['gamma'])
    epochs = int(config['train']['epochs'])
    # load checkpoint if needed/ wanted
    start_epoch = 0
    model = DUFNET_16L().cuda()

    print(model)

    if args.resume:
        ckpt = load_checkpoint(args.path_to_checkpoint)  # custom method for loading last checkpoint
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch']
        optim.load_state_dict(ckpt['optim'])
        print("last checkpoint restored")

    # typically we use tensorboardX to keep track of experiments
    writer = SummaryWriter('runs/train')

    if config['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        print("no exect this optimizer")
        exit()
    scheduler = optim.lr_scheduler.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)            #According to the paper, the learning rate decreases, lr * 0.1 for every 10 epochs
    step = 0

    for epoch in range(start_epoch , epochs+1):
        model.train()
        data_loader = tqdm.tqdm(train_data_loader, desc='Train data loader')
        label_loader = tqdm.tqdm(train_label_loader, desc='Train label loader')
        for data , target in zip(data_loader , label_loader):
            data , target = data[0].cuda(), target[0].cuda()
            output = model(data)
            loss = HuberLoss(output, target)                   # Huber loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()                                         #Learning rate decay
            step += 1
            loss = loss.item()
            # udpate tensorboardX
            writer.add_scalar('train_loss', loss)

        # save checkpoint if needed
        cpkt = {
            'model': model.state_dict(),
            'epoch': epoch,
            'optim': optim.state_dict()
        }
        save_checkpoint(cpkt, './model/model_checkpoint.ckpt')

        if epoch % 10 == 0:
            save_path = os.path.join(pt_dir, 'VSR_DUF.pkl')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'epoch': epoch,
            }, save_path)


