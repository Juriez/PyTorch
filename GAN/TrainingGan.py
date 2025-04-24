import torch  
import torchvision  # Import torchvision for image utilities and datasets.
from torchvision.utils import make_grid  # Import utility to create grids of images for visualization.
import matplotlib.pyplot as plt  # Import Matplotlib for plotting and visualizing images.
import numpy as np  # Import NumPy for numerical operations.

# Import GAN-related components from external files.
from GANOnCifar_10Dataset import epochs, train_loader, device, latent_dimension  # Parameters for dataset and training:
# - epochs: Number of complete passes through the dataset during training.
# - train_loader: DataLoader for batching CIFAR-10 images.

from InitializeGANComponents import optimizer_d, optimizer_g, generator, loss, discriminator  # GAN models and optimizers:
# - optimizer_d: Adam optimizer for discriminator.
# - optimizer_g: Adam optimizer for generator.
# - generator: Neural network for generating fake images.
# - discriminator: Neural network for classifying real vs. fake images.
# - loss: Binary Cross-Entropy Loss function.

# CIFAR-10 image dimensions:
# - Single image = (32, 32, 3) (height, width, RGB channels).
# - Batch of images = (batch_size, 32, 32, 3) (e.g., 32 images in one batch).

for epoch in range(epochs):  # Loop over the number of epochs (full passes through the dataset).
    for i, batch in enumerate(train_loader):  # Loop over batches of images from the DataLoader.
        
        # Convert the batch of real images from list to tensor and move them to the specified device (CPU/GPU).
        real_images = batch[0].to(device)
        
        # Create ground truth labels:
        # - Valid = 1 (real images).
        # - Fake = 0 (fake images).
        valid = torch.ones(real_images.size(0), 1, device=device)  # Tensor of ones for real images.
        fake = torch.zeros(real_images.size(0), 1, device=device)  # Tensor of zeros for fake images.
        
        # Configure input images by moving them to the specified device.
        real_images = real_images.to(device)

        # Train the discriminator.
        optimizer_d.zero_grad()  # Clear previous gradients to avoid accumulation.

        # Sample random noise to use as input for the generator.
        z = torch.randn(real_images.size(0), latent_dimension, device=device)  # Random latent vectors (noise).

        # Generate fake images using the generator.
        fake_images = generator(z)

        # Measure the discriminator's ability to classify real and fake images correctly.
        real_loss = loss(discriminator(real_images), valid)  # Loss for real images.
        fake_loss = loss(discriminator(fake_images.detach()), fake)  # Loss for fake images (detach to avoid gradient computation).
        d_loss = (real_loss + fake_loss) / 2  # Combine the two losses (average).

        # Backward pass and optimize discriminator's parameters.
        d_loss.backward()  # Compute gradients for discriminator based on d_loss.
        optimizer_d.step()  # Update discriminator parameters using the optimizer.

        # Train the generator.
        optimizer_g.zero_grad()  # Clear previous gradients for the generator.

        # Generate fake images using the same random latent vectors z.
        generator_image = generator(z)

        # Compute loss for the generator (tries to fool the discriminator into classifying fake images as real).
        g_loss = loss(discriminator(generator_image), valid)  # Generator wants discriminator to output "real" (valid).

        # Backward pass and optimize generator's parameters.
        g_loss.backward()  # Compute gradients for generator based on g_loss.
        optimizer_g.step()  # Update generator parameters using the optimizer.

        # Progress Reporting:
        if (i+1) % 100 == 0:  # Every 100 batches, print progress.
            print(f'Epoch [{epoch+1}/{epochs}] '
                  f'Batch {i+1}/{len(train_loader)} '
                  f'Discriminator loss: {d_loss.item():.4f} '  # Print discriminator loss.
                  f'Generator loss: {g_loss.item():.4f}'  # Print generator loss.
                  )
            
    # Save generated images after every 10 epochs for monitoring the generator's performance.
    if (epoch+1) % 10 == 0:  # Check if the current epoch is a multiple of 10.
        with torch.no_grad():  # Disable gradient computation for visualization (improves efficiency).
            z = torch.randn(16, latent_dimension, device=device)  # Sample 16 random latent vectors.
            generated = generator(z).detach().cpu()  # Generate images and move them to CPU for visualization.
            grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True)  # Create a grid of 16 images:
            # - nrow=4: 4 images per row.
            # - normalize=True: Normalize pixel values for display.

            plt.imshow(np.transpose(grid, (1, 2, 0)))  # Convert grid tensor (C, H, W) to (H, W, C) for RGB visualization.
            plt.axis("off")  # Remove axes for clean display of images.
            plt.show()  # Display the grid of generated images.
