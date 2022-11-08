import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import typing as ty
from torch import Tensor
import random
import numpy as np

def save_model(model, args):
    torch.save(model.module.state_dict(), args.savepath + "/" + str(args.model) + "_" + str(args.epochs) +".pt")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

def get_optimizer(model) -> optim:
    return torch.optim.Adam(
            model.parameters(),
            lr = 1e-4,
            eps = 1e-8
        )