import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssl
import time
from CnnExampleUsingMnistImage import train_loader, test_loader

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        #fully connected layer
        self.fc1 = nn.Linear(5*5*16,120) #5*5*16 found from after 2nd pooling, 120 is arbitrary number
        self.fc2 = nn.Linear(120,84) #120 is output from previous layer & 84 is arbitrary tobe 120 er choto as decrease hbe
        self.fc3 = nn.Linear(84, 10) #84 is from previous layer & 10 is again arbitrary
        

    def moveForward(self,X):
        #first pass
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2) #2*2 kernel & stride 2

        #2nd pass
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)


        #re-view to flatten it out
        X = X.view(-1, 16*5*5) #-1 used to vary the batch size


        #fully connected layer
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)
    
#create & instance of the model
torch.manual_seed(41)
model = ConvolutionalNetwork()
#print(model)


#loss function optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01) #lr jato smaller hbe, traing time tato greater hbe


start_time = time.time()


#
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    #train
    for b, (X_train, Y_train) in enumerate(train_loader):
        b+=1 #start batches at 1
        Y_prediction = model.moveForward(X_train)
        loss = criterion(Y_prediction, Y_train)
        predicted = torch.max(Y_prediction.data,1)[1] #add the number of correct prediction, & indexed of the first point
        batch_corr = (predicted == Y_train).sum() #got how many correct from this batch, true =1, false = =1, & sum those
        trn_corr += batch_corr #keep track of training


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if b%600 == 0:
            print(f'Epoch: {i} Batch: {b} Loss: {loss.item()}')

    train_losses.append(loss)
    train_correct.append(trn_corr)


    #test

    with torch.no_grad():
        for b, (X_test, Y_test) in enumerate(test_loader):
            Y_value = model.moveForward(X_test)
            predicted = torch.max(Y_value.data,1)[1] 
            tst_corr += (predicted == Y_test).sum() #t=1,f=0

    loss = criterion(Y_value, Y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

current_time = time.time()
total_time = current_time - start_time
print(f'Training time: {total_time/60} minutes!')



#graph the loss at epochs
train_losses = [tl.item() for tl in train_losses]
plt.plot(train_losses, label = "Training loss")
plt.plot(test_losses, label = "Test loss")
plt.title("Loss at epochs")
plt.legend()



#the accuracy graph at the end of each epoch

plt.plot([t/600 for t in train_correct], label = "Training accuracy")
plt.plot([t/100 for t in test_correct], label = "Testing accuracy")
plt.title("Accuracy at the end of each epochs")
plt.legend()