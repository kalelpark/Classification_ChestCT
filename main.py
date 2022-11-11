import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1, 6, 7, 8" 
import yaml
import typing as ty
import argparse
from dataset import *
import torch
from train import model_train
from eval import model_infer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type = str, required = True)        # train
    parser.add_argument("--seed", type = int, required = True)        # train
    parser.add_argument("--model", type = str, required = True)
    parser.add_argument("--epochs", type = int, required = True)
    parser.add_argument("--batchsize", type = int, required = True)
    parser.add_argument("--savepath", type = str, required = True)
    args = parser.parse_args()
    
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.action == "train":
        model_train(args)
    else:
        model_infer(args)