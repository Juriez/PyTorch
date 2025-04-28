import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import ssl

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set device: Use GPU if available; otherwise, use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 'cuda' means GPU; 'cpu' is the fallback


# Define image transformations for preprocessing the MNIST dataset
transform = transforms.Compose({
    transforms.ToTensor() # Convert images to PyTorch tensors for neural network compatibility
})

# SSL Certificate Handling: Avoid SSL-related issues during dataset download
ssl._create_default_https_context = ssl._create_unverified_context   # Create an unverified SSL context

# Load the MNIST dataset (training data)
train_data = datasets.MNIST(root = '/mnistData', train=True, download=True, transform= transform)
# 'root': Directory to save dataset files
# 'train=True': Load the training split of MNIST
# 'download=True': Download the dataset if not present locally
# 'transform': Apply preprocessing transformations (defined above)

# Create a DataLoader for batching and shuffling training data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
# 'batch_size=8': Process data in batches of 8 for training efficiency
# 'shuffle=True': Shuffle the data to improve training by introducing randomness

