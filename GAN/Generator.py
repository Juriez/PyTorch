import torch.nn as nn

#define generator
class Generator(nn.Module):
    def __init__(self, latent_dimension):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dimension, 128*8*8),
            nn.ReLU(),
            nn.Unflatten(1,(128,8,8)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.78),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64, momentum=0.78),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self,z):
        img = self.model(z)  #z = latent vector(noise), transform the vactor into image
        return img
