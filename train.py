from generator import *
from discriminator import *
from custom_dataloader import ImageDataLoader
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

number_of_epochs = 50
batch_size = 4
shape = (256, 256)
cuda = torch.cuda.is_available()

generator = Generator()
discriminator = Discriminator(input_shape=(3, *shape))
feature_extractor = FeatureExtractor()
feature_extractor.eval()
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()
generator = generator.cuda()
discriminator = discriminator.cuda()
feature_extractor = feature_extractor.cuda()
criterion_GAN = criterion_GAN.cuda()
criterion_content = criterion_content.cuda()


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

dataloader = DataLoader(ImageDataLoader("data/img_align_celeba", shape=shape),batch_size=batch_size,shuffle=False,num_workers=8)


for epoch in range(num_of_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(torch.cuda.FloatTensor))
        imgs_hr = Variable(imgs["hr"].type(torch.cuda.FloatTensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()
    
    # Save model checkpoints every in every iteration
    torch.save(generator.state_dict(), "saved_model/generator_%d.pth" % epoch)
    torch.save(discriminator.state_dict(), "saved_model/discriminator_%d.pth" % epoch)