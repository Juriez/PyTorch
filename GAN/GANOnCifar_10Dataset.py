import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import ssl

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])


# SSL Certificate Handling
ssl._create_default_https_context = ssl._create_unverified_context

#load the cifar-10 dataset
#train data
train_data = datasets.CIFAR10(root='/cifar10_data', train=True, download= True, transform= transform)
train_loader = torch.utils.data.DataLoader(train_data, \
                                batch_size=32, shuffle=True)


#define gan parameters
latent_dimension = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
epochs = 10


