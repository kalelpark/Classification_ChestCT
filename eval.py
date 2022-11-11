import os
import torch
import torch.nn as nn
from torch import LongTensor
import pandas as pd 
import typing as ty
import yaml
import numpy as np
from dataset import *
from utils import *
from models import common

import wandb
import torch.optim as optim
from metric import *

def model_infer(args):
    seed_everything(args.seed)

    # Ensemble Code
    # dl_test = get_dataloader(args)
    # # temp = next(iter(dl_test))
    # submit = pd.read_csv("data/sample_submission.csv")
    # model_path = [  "ensemble_storage/33_efficientnet-b6_100.pt", "ensemble_storage/3_efficientnet-b6_100.pt",
    #                 "ensemble_storage/333_efficientnet-b6_100.pt", "ensemble_storage/3333_efficientnet-b6_100.pt",
    #                 "ensemble_storage/33333_efficientnet-b6_100.pt", "ensemble_storage/333333_efficientnet-b6_100.pt"]
    

    # 
    # ensemble_submit = torch.zeros(999, 5)
    # for path in model_path:
    #     model = common.load_model(args)
    #     model.load_state_dict(torch.load(path))          
    #     model.to(args.device)
    #     model = nn.DataParallel(model)

    #     array = list()
    #     model.eval()
    #     with torch.no_grad():
    #         for img in dl_test:
    #             img = img.float().to(args.device)
    #             model_pred = model(img)
    #             model_pred = model_pred.squeeze(1).to('cpu')
    #             array.append(model_pred)
            
    #         ensemble_submit += torch.concat(array, axis = 0)
    
    # ensemble_submit /= len(model_path)
    # model_preds = ensemble_submit.argmax(1).detach().cpu().numpy().tolist()

    # Single Code
    dl_test = get_dataloader(args)
    temp = next(iter(dl_test))
    submit = pd.read_csv("data/sample_submission.csv")
    model_path = "savemodel/efficientnet-b0_100.pt"

    model = common.load_model(args)
    model.load_state_dict(torch.load(model_path))          
    model.to(args.device)
    model = nn.DataParallel(model)

    model.eval()
    model_preds = []

    with torch.no_grad():
        for img in dl_test:
            img = img.float().to(args.device)
            model_pred = model(img)
            model_pred = model_pred.squeeze(1).to('cpu')
                
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()

    {'ILD': 0, 'Lung_Cancer': 1, 'Normal': 2, 'pneumonia': 3, 'pneumothorax': 4}
    {'ILD': 0, 'Lung_Cancer': 1, 'Normal': 2, 'pneumonia': 3, 'pneumothorax': 4}

    submit["result"] = model_preds
    submit["result"] = submit["result"].replace(0, "ILD")
    submit["result"] = submit["result"].replace(1, "Lung_Cancer")
    submit["result"] = submit["result"].replace(2, "Normal")
    submit["result"] = submit["result"].replace(3, "pneumonia")
    submit["result"] = submit["result"].replace(4, "pneumothorax")

    submit["result"] = submit["result"].replace("Normal", 0)
    submit["result"] = submit["result"].replace("Lung_Cancer", 1)
    submit["result"] = submit["result"].replace("ILD", 2)
    submit["result"] = submit["result"].replace("pneumonia", 3)
    submit["result"] = submit["result"].replace("pneumothorax", 4)
    
    submit.to_csv('submission/ensemble.csv', index=False)
