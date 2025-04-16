import torch
import torch.nn as nn #nn: neural network
import torch.nn.functional as F
import ssl
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

#create a model class that inherits nn.Module

class Model(nn.Module):
    #Input layer(4 features of the flower)-->HiddenLayer1(number of neurons)-->H2(n)-->outputLayer(3 classes of iris flowers)

    def __init__(self, input_features=4, h1=8, h2=9, output_features=3):
       super().__init__()  #instanciate the nn.Module
       self.fc1 = nn.Linear(input_features,h1) #flow connection between input features & hidden layer 1
       self.fc2 = nn.Linear(h1,h2) #flow connection between hidden layer 1 & 2
       self.final = nn.Linear(h2,output_features) #flow connection between hidden layer2 & output features

    def moveForward(self,x):
        
        x = F.relu(self.fc1(x))  #relu: rectified linear unit
        x = F.relu(self.fc2(x))
        x = self.final(x)
        return x
    
#pick a manual seed for randomization
torch.manual_seed(41)
model = Model()

# SSL Certificate Handling
ssl._create_default_https_context = ssl._create_unverified_context


url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)
print(my_df)