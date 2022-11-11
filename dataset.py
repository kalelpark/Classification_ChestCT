from transforms import get_transform
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
from PIL import Image


trainset = torchvision.datasets.ImageFolder(root = "data/Train",
                                            transform = get_transform("train")
                                        )

validset = torchvision.datasets.ImageFolder(root = "data/Validation",
                                            transform = get_transform("valid")
                                        )

# {'ILD': 0, 'Lung_Cancer': 1, 'Normal': 2, 'pneumonia': 3, 'pneumothorax': 4}

image_list = os.listdir("data/ct_new")
image_list.sort()

class customDataset(Dataset):
    def __init__(self, image_listed):
        self.image_list = image_listed
        self.transforms = get_transform("test")
        self.paths = "data/ct_new/"
    
    def __getitem__(self, x):
        image_path = self.image_list[x]
        image = cv2.imread(os.path.join(self.paths, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transforms is not None:
            image = self.transforms(image)
        
        return image

    def __len__(self):
        return len(self.image_list)


def get_dataloader(args):
    if args.action == "train":
        trainloader = DataLoader(   trainset, shuffle = True, 
                                    batch_size = args.batchsize, pin_memory = True,)

        validloader = DataLoader(   validset, shuffle = True, 
                                    batch_size = args.batchsize, pin_memory = True,)
    
        return trainloader, validloader
    else:
        test_dataset = customDataset(image_list)
        testloader = DataLoader(test_dataset, batch_size = args.batchsize, pin_memory = True)
        return testloader
