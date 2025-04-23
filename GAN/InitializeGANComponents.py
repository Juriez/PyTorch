import torch.nn as nn
import torch.optim as optim

from Generator import Generator
from Discriminator import Discriminator
from  GANOnCifar_10Dataset import latent_dimension, device, lr, beta1, beta2

#define & initialize generator & discriminator
generator = Generator(latent_dimension).to(device)
discriminator = Discriminator().to(device)

#loss function
loss = nn.BCELoss()

#optimizers
optimizer_g = optim.Adam(generator.parameters()\
                         , lr = lr, betas= (beta1, beta2))
optimizer_d = optim.Adam(discriminator.parameters()\
                         , lr = lr, betas=(beta1,beta2))

