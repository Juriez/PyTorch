import torch
import numpy as np

np1 = np.random.rand(3,4)
print("numpy array: ", np1)

tensor_2d = torch.randn(3,4)
print(tensor_2d.dtype)
print("2d tensor: ",tensor_2d)


tensor_3d = torch.randn(2,3,4)
print("3d tensor: ", tensor_3d)

my_tensor = torch.tensor(np1)
print("Numpy in tensor: ", my_tensor)