import torch.nn as nn  

# Define the generator class that inherits from nn.Module
class Generator(nn.Module):
    def __init__(self, latent_dimension):  # Initialization function for the generator
        super(Generator, self).__init__()  # Call the superclass constructor (nn.Module)

        # Define the generator's neural network using nn.Sequential
        self.model = nn.Sequential(
            nn.Linear(latent_dimension, 128*8*8),  # Fully connected layer mapping latent vector (z) to feature space.
            nn.ReLU(),  # Apply ReLU activation to introduce non-linearity.
            nn.Unflatten(1, (128, 8, 8)),  # Reshape the output of the previous layer into a tensor with shape (128, 8, 8).
            nn.Upsample(scale_factor=2),  # Upsample the feature map by a factor of 2 (e.g., increase dimensions).
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Apply 2D convolution with 128 filters, kernel size 3x3, padding 1.
            nn.BatchNorm2d(128, momentum=0.78),  # Apply Batch Normalization to stabilize training and improve convergence.

            #Apply similar opperations for better performance
            nn.ReLU(),  
            nn.Upsample(scale_factor=2),  
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Apply 2D convolution reducing the filters from 128 to 64.
            nn.BatchNorm2d(64, momentum=0.78), 
            nn.ReLU(), 
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Final convolution to produce 3-channel output (RGB image).
            nn.Tanh(),  # Apply Tanh activation to scale output pixel values to the range [-1, 1].
        )

    # Define the forward pass of the generator
    def forward(self, z):  # Input: latent vector (noise) z
        img = self.model(z)  # Pass z through the defined network to transform it into an image
        return img  # Output: generated image
