# Import required libraries
import torch
import torch.optim as optim
import torch.nn as nn

from AddingNoiseToData import corrupt
from LoadAndPreprocessDataset import train_loader, device  # Import train_loader and device from the data loader file
from BasicUNet import BasicUNet  # Import the BasicUNet model

# Define the training function
def train_unet(model, dataloader, epochs=5, lr=0.001):
    # Set the model to training mode
    model.train()

    # Define the optimizer (Adam in this case) and loss function (Mean Squared Error Loss)
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer with learning rate lr
    loss = nn.MSELoss()  # Mean Squared Error Loss function

    # Training loop for the specified number of epochs
    for epoch in range(epochs):
        epoch_loss = 0.0  # Initialize loss accumulator for the epoch

        for batch_idx, (x, y) in enumerate(dataloader):  # Iterate over batches of data
            # Move the data to the specified device (GPU/CPU)
            x = x.to(device)  # Input data
            y = y.to(device)  # Target labels (for supervised tasks, but ignored in this case)

            # Zero out the gradients from the previous step
            optimizer.zero_grad()

            # Forward pass: Get model predictions
            corrupted_x = corrupt(x, torch.rand(x.shape[0]).to(device))  # Add random noise to input data
            output = model(corrupted_x)  # Pass the corrupted input through the U-Net

            # Compute the loss (difference between original input and reconstructed output)
            losses = loss(output, x)  # Compare reconstructed output to the original data

            # Backward pass: Compute gradients
            losses.backward()

            # Update model weights based on gradients
            optimizer.step()

            # Accumulate the loss for this batch
            epoch_loss += losses.item()
            
            # Progress Reporting:
            if (batch_idx+1) % 100 == 0: # Every 100 batches, print progress
                print(f"Epoch [{epoch + 1}/{epochs}] "
                f"Batch {batch_idx+1}/{len(train_loader)} " 
                f"Loss: {epoch_loss / len(dataloader):.4f} ")
                

# Instantiate the BasicUNet model and move it to the device
model = BasicUNet().to(device)

# Train the model using the train loader
train_unet(model, train_loader, epochs=5, lr=0.001)  # Train for 10 epochs with a learning rate of 0.001
