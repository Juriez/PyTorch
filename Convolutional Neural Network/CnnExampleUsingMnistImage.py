import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssl

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix


# SSL Certificate Handling
ssl._create_default_https_context = ssl._create_unverified_context


#converts mnist image files into 4 dimensions tensor(#of images, height, width, color, channel)

transforms = transforms.ToTensor()

#Train data
train_data = datasets.MNIST(root='/cnn_data', train= True, download= True, transform= transforms)



#Test data
test_data = datasets.MNIST(root='/cnn_data', train= False, download= True, transform= transforms)
# print("Train Data: \n")
# print(train_data)

# print("Test Data: \n")
# print(test_data)

#create a small batch size for images , here it's 10
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)


#Define CNN model
conv1 = nn.Conv2d(1,6,3,1)  #(one image, node on 2nd layer, kernel size, stride)
conv2 = nn.Conv2d(6, 16,3,1) #(2nd layer er output 6 so erporer layer e input 6, 3rd layer er node 16)


#grab one MNIST image
for i, (X_train, Y_train) in enumerate(train_data):
    break

print(X_train.shape)
x = X_train.view(1,1,28,28)

#perform te first convolution
x = F.relu(conv1(x))
print("after convolution:\n")
print(x.shape) # 1 is image, 6 is filters we asked for, 26*26


#pass through the pooling layer]
x = F.max_pool2d(x,2,2)  #kernel of 2, stride of 2
print("after pooling:\n")
print(x.shape) #in output 13 is = 26/2(26 convolution er output)


#perform the 2nd convolution
x = F.relu(conv2(x))
print("after 2nd convolution:\n")
print(x.shape) #we didn't set padding so we lose 2 pixels around the outside of the image


#2nd pooling layer
x = F.max_pool2d(x,2,2)
print("after 2nd pooling:\n")
print(x.shape)
