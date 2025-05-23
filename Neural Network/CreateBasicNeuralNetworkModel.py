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
        #relu: rectified linear unit, F.relu introduces non-linearity and ensures that negative values are set to 0 
        x = F.relu(self.fc1(x)) #input vlaue passed to the h1 through fc1 
        x = F.relu(self.fc2(x)) #the output from h1 passed to the h2 through fc2
        x = self.final(x) #processed data from the h2 passed to the output layer through final
        return x
    
#pick a manual seed for randomization
torch.manual_seed(41)
model = Model()

# SSL Certificate Handling
ssl._create_default_https_context = ssl._create_unverified_context


url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)
print(my_df)
# print("my_df head: \n")
# print(my_df.head()) #show first 5 rows
# print("my_df tail: \n")
# print(my_df.tail()) #show last 5 rows

#if we want to replace any column values

my_df['species'] = my_df['species'].replace('setosa', 0.0) #replace the characteristics setosa with 0.0
my_df['species'] = my_df['species'].replace('versicolor', 1.0) #replace the characteristics versicolor with 1.0
my_df['species'] = my_df['species'].replace('virginica', 2.0) #replace the characteristics virginica with 2.0
# print("my_df after replacing characteristics: \n")
# print(my_df)


#Train Test split

X = my_df.drop('species', axis = 1)
Y = my_df['species']
print("X :\n", X)
print("Y :\n", Y)
#convert x,y to numpy arrays
X = X.values
Y = Y.values
# print("X :\n", X)
# print("Y :\n", Y)

#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=41) #test_size =0.2, so train _size must be 0.8

#convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

#convert Y features to long tensors
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

#set the criteria of the model to measure errors, how far of the predictions from the data
criterion = nn.CrossEntropyLoss()
#choose Adam optimizer, lr = learning rate(if error doesn't go down after a bunch of iterations(epochs), lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

# print(model.parameters)

#train our model
#Epochs? (one run thru all the training data in our network)

epochs = 100
losses = []
for i in range(epochs):
    #go forward & predict the result
    Y_prediction = model.moveForward(X_train)

    #measure the error, will be high at first then exponentially will decreases
    loss = criterion(Y_prediction, Y_train) #predicted value vs trained value

    #keep track the losses
    losses.append(loss.item()) #loss.detach().numpy(

    #print every 10 epochs
    if i%10 == 0:
        print(f'Epoch: {i} and loss: {loss}')
    
    #take the error rate of forward propagation and feed it back

    #thru the network to fine tune the weights

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #lets plot the graph
plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel("epoch")
plt.title("Training Loss Over Epochs")
plt.show()

#evaluate Model on test data set(validation basically)
with torch.no_grad():
    Y_evaluation = model.moveForward(X_test) #x_test are features from our test set & y_evaluation will be the predictions
    loss = criterion(Y_evaluation, Y_test)

#print(loss)

correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        Y_value = model.moveForward(data)
        m = Y_value.argmax().item()

        if Y_test[i] == 0:
            x = 'setosa'
        elif Y_test[i] == 1:
            x = 'versicolor'
        elif Y_test[i] == 2:
            x = 'virginica'

         #identify correct or not
        if m == Y_test[i]:
            correct +=1
        
        #m
        if m == 0:
            m = 'setosa'
        elif m == 1:
            m = 'versicolor'
        elif m == 2:
            m = 'virginica'
        
        #will show the type of flowers
        print(f'{i+1}.) {str(Y_value)} \t {x} \t {m}')

       
    print(f'Got {correct} correct!')


#evaluate new data on the network

new_iris = torch.tensor([5.9,3.0,5.1,1.8]) #4 values of X
with torch.no_grad():
    print(model.moveForward(new_iris))
    

#save the NN model
torch.save(model.state_dict(), 'my_iris_model')

#load the saved model
new_model = Model()
print(new_model.load_state_dict(torch.load('my_iris_model')))

#make sure it loaded correctly
print(new_model.eval())
