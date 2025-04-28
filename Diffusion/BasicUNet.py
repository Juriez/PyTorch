import torch.nn as nn
from LoadAndPreprocessDataset import device

class BasicUNet(nn.Module):  # Define the BasicUNet class, inheriting from nn.Module.
    def __init__(self, in_channels=1, out_channels=1):  # Initialize the class with input and output channels.
        super().__init__()  # Call the parent class constructor.
        # Define the downsampling path with a series of convolutional layers.
        self.down_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),  # First convolution: 1 input channel -> 32 output channels.
            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # Second convolution: 32 -> 64 channels.
            nn.Conv2d(64, 64, kernel_size=5, padding=2),  # Third convolution: 64 -> 64 channels.
        ])

        # Define the upsampling path with a series of convolutional layers.
        self.up_layers = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),  # First upsampling convolution: 64 -> 64 channels.
            nn.Conv2d(64, 32, kernel_size=5, padding=2),  # Second upsampling convolution: 64 -> 32 channels.
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),  # Final convolution: 32 -> output channels.
        ])

        self.act = nn.SiLU()  # Define the activation function (SiLU, also called Swish).
        self.downscale = nn.MaxPool2d(2)  # Define downscaling using max pooling with a 2x2 window.
        self.upscale = nn.Upsample(scale_factor=2)  # Define upscaling to double the spatial dimensions.

    def forward(self, x):  # Define the forward pass of the network.
        h = []  # Initialize a list to store feature maps for skip connections.

        # Downsampling path
        for i, l in enumerate(self.down_layers):  # Iterate through the downsampling layers.
            x = self.act(l(x))  # Apply convolution followed by the activation function.
            if i < 2:  # Store the feature map only from the first two layers for skip connections.
                h.append(x)  # Save the feature map in the list.
                x = self.downscale(x)  # Downscale the feature map using max pooling.

        # Upsampling path
        for i, l in enumerate(self.up_layers):  # Iterate through the upsampling layers.
            if i > 0:  # Skip the first layer for upscaling (it processes the bottleneck features).
                x = self.upscale(x)  # Upscale the feature map by doubling its spatial dimensions.
                x += h.pop()  # Add (merge) the corresponding feature map from the skip connections.
            x = self.act(l(x))  # Apply convolution followed by the activation function.

        return x  # Return the reconstructed output.

net = BasicUNet().to(device)  # Instantiate the BasicUNet class and move it to the specified computation device (GPU/CPU).
