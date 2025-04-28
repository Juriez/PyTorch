import torch

from LoadAndPreprocessDataset import train_loader

# Define a function to apply noise to the input data (x) with a specified amount of corruption.
def corrupt(x, amount):
    # Generate random noise tensor with the same shape as the input data x.
    noise = torch.rand_like(x)
    # Reshape the amount tensor to ensure compatibility for broadcasting during computation.
    amount = amount.view(-1, 1, 1, 1)
    # Combine the original data with the noise based on the corruption amount. Interpolates between x and noise.
    return x * (1 - amount) + noise * amount

# Retrieve the next batch of input data (x) and corresponding labels (y) from the data loader.
x, y = next(iter(train_loader))
# Generate a tensor of evenly spaced values between 0 and 1 to represent the corruption levels for the batch.
amount = torch.linspace(0, 1, x.shape[0])
# Apply the corrupt function to the input data using the generated corruption levels.
noised_x = corrupt(x, amount)
