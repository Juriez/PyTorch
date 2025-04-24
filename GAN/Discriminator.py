import torch.nn as nn  # Import PyTorch's neural network module for defining layers and components.

# Define the discriminator class that inherits from nn.Module
class Discriminator(nn.Module):
    def __init__(self):  # Initialization function for the discriminator
        super(Discriminator, self).__init__()  # Call the superclass constructor (nn.Module)

        # Define the discriminator's neural network using nn.Sequential
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Convolutional layer:
            # - Input: 3-channel image (e.g., RGB).
            # - Output: 32 feature maps with kernel size 3x3, stride 2 (downsampling), padding 1.
            nn.LeakyReLU(0.2),  # Leaky ReLU activation:
            # - Allows small gradients when the input is negative.
            # - Negetive values(Xi) are replced with (0.2*Xi)
            nn.Dropout(0.25),  # Dropout layer:
            # - Randomly sets 25% of neurons to zero during training to prevent overfitting.

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            # - Input: 32 feature maps.
            nn.ZeroPad2d((0, 1, 0, 1)),  # Zero padding:
            # - Adds padding to the right and bottom of the feature map.
            # - Ensures spatial dimensions are consistent for specific operations.

            nn.BatchNorm2d(64, momentum=0.82),  # Batch normalization for 64 feature maps:
            # - Normalizes the feature maps to stabilize learning.
            # - Momentum controls how quickly moving averages adjust.
            nn.LeakyReLU(0.25), 
            nn.Dropout(0.25),  # Dropout layer (25% dropout again).


            #Apply similar opperations for better performance
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(128, momentum=0.82), 
            # - Stabilizes learning by normalizing outputs.
            nn.LeakyReLU(0.2),  
            nn.Dropout(0.25),  

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256, momentum=0.8), 
            nn.LeakyReLU(0.25), 
            nn.Dropout(0.25),  # Dropout layer (25% dropout again).

            nn.Flatten(),  # Flatten layer:
            # - Converts the 4D feature map tensor (batch_size, 256, 5, 5) into a 2D tensor (batch_size, 256*5*5).

            nn.Linear(256*5*5, 1),  # Fully connected layer:
            # - Maps flattened feature vector to a single output value (validity score).

            nn.Sigmoid(),  # Sigmoid activation:
            # - Ensures the output validity score is in the range [0, 1].
        )

    # Define the forward pass of the discriminator
    def forward(self, img):  # Input: image
        validity = self.model(img)  # Pass the input image through the discriminator model.
        return validity  # Output: validity score (0 = fake, 1 = real)
