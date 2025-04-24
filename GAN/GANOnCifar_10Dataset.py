import torch  # Import PyTorch
import numpy as np  # Import NumPy for numerical operations.
import torch.nn as nn  # Import PyTorch's neural network module to define layers like Linear or Conv2D.
import torch.optim as optim  # Import optimizers like Adam and SGD for training models.
import ssl  # Import SSL to manage secure connections (used for dataset downloads).

from torchvision import datasets, transforms  # Import torchvision for accessing datasets and preprocessing tools.
from torch.utils.data import DataLoader  # Import DataLoader for efficient data loading and batching.

# Set device: Use GPU if available; otherwise, use CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'cuda' means GPU; 'cpu' is the fallback.

# Define image transformations for preprocessing the CIFAR-10 dataset.
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors for neural network compatibility.
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values to the range [-1, 1].
])

# SSL Certificate Handling: Avoid SSL-related issues during dataset download.
ssl._create_default_https_context = ssl._create_unverified_context  # Create an unverified SSL context.

# Load the CIFAR-10 dataset (training data).
train_data = datasets.CIFAR10(root='/cifar10_data', train=True, download=True, transform=transform)  
# 'root': Directory to save dataset files.
# 'train=True': Load the training split of CIFAR-10.
# 'download=True': Download the dataset if not present locally.
# 'transform': Apply preprocessing transformations (defined above).

# Create a DataLoader for batching and shuffling training data.
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)  
# 'batch_size=32': Process data in batches of 32 for training efficiency.
# 'shuffle=True': Shuffle the data to improve training by introducing randomness.

# Define GAN parameters.
latent_dimension = 100  # Size of the random latent vector (input to the generator).
lr = 0.0002  # Learning rate for optimizers; controls the step size during parameter updates.
beta1 = 0.5  # Beta1 value for Adam optimizer; controls momentum for the first moment estimate.
beta2 = 0.999  # Beta2 value for Adam optimizer; controls momentum for the second moment estimate.
epochs = 10  # Number of epochs (full passes through the training dataset).
