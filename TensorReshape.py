import torch

#reshape
my_tensor = torch.arange(10)
print("Vector tensor: ", my_tensor)

my_tensor = my_tensor.reshape(2,5)
print("Reshaped Tensor 2*5: ", my_tensor)

my_tensor = my_tensor.reshape(5,2)
print("Reshaped tensor 5*2: ", my_tensor)

#reshape if we don't know the number of elements using -1

my_tensor2 = torch.arange(15)
print("Tensor 2: ", my_tensor2)

my_tensor2 = my_tensor2.reshape(3,-1) #or we can use -1 as the first value like(-1,3) if we want to fixed the column number
print("Reshaped Tensor: ", my_tensor2)

#it will show an error 
# my_tensor2 = my_tensor2.reshape(4,-1)
# print(my_tensor2)



#Slices

my_tensor3 = torch.arange(20)

#grab slice
my_tensor4 = my_tensor3.reshape(10,2)
print(my_tensor4)
print("slices: ", my_tensor4[:,1]) #show 2nd column

my_tensor4 = my_tensor3.reshape(4,5)
print("My tensor 4: ", my_tensor4)
print("3rd & 5th column's elements: ", my_tensor4[:,:,2,:,:])