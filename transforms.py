import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


def get_transform(action):
    if action == "train":
        return transforms.Compose([
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

