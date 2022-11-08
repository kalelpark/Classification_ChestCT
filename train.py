import os
import torch
import torch.nn as nn
from torch import LongTensor
import typing as ty
import yaml
import numpy as np
from dataset import *
from utils import *
from models import common

import wandb
import torch.optim as optim
from metric import *


def model_train(args):
    seed_everything(args.seed)
    model = common.load_model(args)

    dl_train, dl_valid = get_dataloader(args)
    
    model.to(args.device)
    model = nn.DataParallel(model)
    loss_fn = nn.CrossEntropyLoss().to(args.device)
    print("load model..")
    optimizer = get_optimizer(model)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 20, T_mult = 2, eta_min = 0)
    # python main.py --action train --seed 0 --model efficientnet-b0 --epochs 50 --batchsize 64 --savepath savemodel

    wandb.init( name = args.model + "_" + str(args.epochs), 
                project = "CT_Image_Competition" , reinit = True)

    best_score = 1e9
    for epoch in range(10000):
        train_loss, valid_loss = [], []
        pred_label, true_label = [], []
        print(f"{str(epoch + 1)} / {str(args.epochs)}")
        model.train()
        for img, y in dl_train:
            # break        # just train
            img, y = img.float().to(args.device) , y.to(args.device)
            output = model(img)
            loss = loss_fn(output, y)
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss.append(loss.item())
        scheduler.step()

        model.eval()
        for img, y in dl_valid:
            img, y = img.float().to(args.device) , y.to(args.device)
            output = model(img)
            loss = loss_fn(output, y)

            pred_label += output.argmax(1).detach().cpu().numpy().tolist()
            true_label += y.cpu().numpy().tolist()

            valid_loss.append(loss.item())

        if best_score > np.mean(valid_loss):
            best_score = np.mean(valid_loss)
            save_model(model, args)
        
        wandb.log({
            "train_loss" : np.mean(train_loss),
            "valid_loss" : np.mean(valid_loss),
            "valid_accuracy" : get_accuracy(true_label, pred_label),
            "valid_f1_score" : get_f1_score(true_label, pred_label)
        })


            
            
    


    



