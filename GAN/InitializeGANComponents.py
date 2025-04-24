import torch.nn as nn 
import torch.optim as optim 

from Generator import Generator  # Import the previously defined Generator class.
from Discriminator import Discriminator  # Import the previously defined Discriminator class.
from GANOnCifar_10Dataset import latent_dimension, device, lr, beta1, beta2  # Import parameters from GANOnCifar_10Dataset:
# - latent_dimension: Size of the random noise vector (z).
# - device: Specifies whether to use GPU or CPU for computation.
# - lr: Learning rate for optimizers.
# - beta1, beta2: Hyperparameters for Adam optimizer.

# Define and initialize the generator.
generator = Generator(latent_dimension).to(device)  # Instantiate the generator and move it to the specified device.
# - to(device): Moves the generator to GPU or CPU based on availability.

# Define and initialize the discriminator.
discriminator = Discriminator().to(device)  # Instantiate the discriminator and move it to the specified device.
# - No additional arguments required; the discriminator works directly on input images.

# Define the loss function.
loss = nn.BCELoss()  # Binary Cross-Entropy Loss:
# - Measures how well the discriminator differentiates real and fake images.
# - Used by both the generator (to fool the discriminator) and the discriminator (to classify correctly).

# Define the optimizer for the generator.
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))  
# - generator.parameters(): Specifies the generator's learnable parameters to optimize.
#   - beta1=0.5: Momentum for first moment estimates.
#   - beta2=0.999: Momentum for second moment estimates.

# Define the optimizer for the discriminator.
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))  
# - discriminator.parameters(): Specifies the discriminator's learnable parameters to optimize.

