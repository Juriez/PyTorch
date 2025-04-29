import torch
import matplotlib.pyplot as plt  # Import matplotlib for visualization

from LoadAndPreprocessDataset import train_loader, device
from AddingNoiseToData import corrupt
from Training import model

# Define a function to visualize images
def show_images(original, noisy, reconstructed):
    batch_size = original.shape[0]  # Number of images in the batch
    
    plt.figure(figsize=(15, 5))  # Set the figure size
    for i in range(batch_size):
        # Plot the original image
        plt.subplot(3, batch_size, i + 1)
        plt.imshow(original[i].squeeze().cpu().numpy(), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # Plot the noisy image
        plt.subplot(3, batch_size, batch_size + i + 1)
        plt.imshow(noisy[i].squeeze().cpu().numpy(), cmap="gray")
        plt.title("Noisy")
        plt.axis("off")

        # Plot the reconstructed image
        reconstructed_normalized = (reconstructed[i].squeeze().cpu().numpy() - reconstructed[i].min().item()) / (
            reconstructed[i].max().item() - reconstructed[i].min().item()
        )  # Normalize reconstructed images for clear visualization
        plt.subplot(3, batch_size, 2 * batch_size + i + 1)
        plt.imshow(reconstructed_normalized, cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

    plt.tight_layout()
    plt.show()  # Display the images

# Example: Visualize a batch of images
x, _ = next(iter(train_loader))  # Get a batch of original images
x = x.to(device)  # Move the batch to the same device as the model

# Corrupt the images with noise
amount = torch.rand(x.shape[0]).to(device)  # Random noise level
corrupted_x = corrupt(x, amount)  # Add noise to the images

# Reconstruct the noisy images using the trained model
model.eval()  # Switch the model to evaluation mode
with torch.no_grad():  # Disable gradient computation
    reconstructed_x = model(corrupted_x)

# Show the original, noisy, and reconstructed images
show_images(x, corrupted_x, reconstructed_x)
