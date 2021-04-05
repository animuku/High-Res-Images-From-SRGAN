import os
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from generator import *
from discriminator import *
from custom_dataloader import ImageDataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2

cuda = torch.cuda.is_available()
shape = (256, 256)
generator = Generator()
discriminator = Discriminator(input_shape=(3,*shape))
generator = generator.cuda()
generator.load_state_dict(torch.load("saved_models/generator_3.pth")
dataloader = DataLoader(
    ImageDataLoader("test_data/test/", shape=(256,256)),
    batch_size=1,
    shuffle=True,
    num_workers=8,
)

for i,imgs in enumerate(dataloader):
    imgs_lr = Variable(imgs["lr"].type(torch.cuda.FloatTensor))
    gen_hr = generator(imgs_lr)
    save_image(gen_hr,"images%d.jpg"%i,normalize=False)
