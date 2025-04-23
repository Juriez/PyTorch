import torch
import torchvision
from torchvision.utils import make_grid 
import matplotlib.pyplot as plt
import numpy as np

from GANOnCifar_10Dataset import epochs, train_loader,device, latent_dimension
from InitializeGANComponents import optimizer_d, optimizer_g, generator,loss, discriminator

#The discriminator is trained to differentiate between real and fake images
#The generator is trained to produce realistic images that fool the discriminator

for epoch in range(epochs):
    for i, batch in enumerate(train_loader):
        #convert list to tensor
        real_images = batch[0].to(device)
        valid = torch.ones(real_images.size(0), 1, device=device)
        fake = torch.ones(real_images.size(0), 1, device=device)
        
        #configure input
        real_images = real_images.to(device)

        #train discriminator
        optimizer_d.zero_grad()

        #sample noise as genrator input
        z = torch.randn(real_images.size(0), latent_dimension, device=device)
        fake_images = generator(z)


        #lets measure the discriminator's ability to find out real/fake image
        real_loss = loss(discriminator\
                         (real_images), valid)
        fake_loss = loss(discriminator\
                         (fake_images.detach()), fake)
        d_loss = (real_loss + fake_loss)/2

        #backward pass & optimize
        d_loss.backward()
        optimizer_d.step()


        #train generator
        optimizer_g.zero_grad()
        generator_image = generator(z)
        g_loss = loss(discriminator(generator_image), valid)

        #backward pass & optimize
        g_loss.backward()
        optimizer_g.step()


        #progress
        if (i+1) % 100 ==0:
            print(f'Epoch [{epoch+1}/{epochs}]\
                  Batch {i+1}/{len(train_loader)} '
                  f'Discriminator loss: {d_loss.item():.4f} '
                  f'Generator loss: {g_loss.item():.4f}'
                  )
            
    #save generated images for every epoch
    if(epoch+1) %10 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dimension, device=device)
            generated = generator(z).detach().cpu()
            grid = torchvision.utils.make_grid(generated,\
                                               nrow=4, normalize=True)
            plt.imshow(np.transpose(grid, (1,2,0)))
            plt.axis("off")
            plt.show()
            




