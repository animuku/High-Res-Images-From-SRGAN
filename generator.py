import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19

class upsample(nn.Module):
    def __init__(self,channels,scale):
        super(upsample, self).__init__()
        self.conv = nn.Conv2d(channels,channels*scale**2,kernel_size=3,padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.relu = nn.ReLU()
    
    def forward(self,X):
        X = self.conv(X)
        X = self.pixel_shuffle(X)
        X = self.relu(X)
        return X

class res_block(nn.Module):
    def __init__(self,channels):
        super(res_block, self).__init__()
        self.conv_1 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.batchnorm_1 = nn.BatchNorm2d(channels)
        self.relu = nn.PReLU()
        self.conv_2 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(channels)

    def forward(self,x):
        result = self.conv_1(x)
        result = self.batchnorm_1(result)
        result = self.relu(result)
        result = self.conv_2(result)
        result = self.batchnorm_2(result)
        return result + x #connecting input with result, thus creating a skip connection 


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(Generator, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())
        self.block2 = res_block(64)
        self.block3 = res_block(64)
        self.block4 = res_block(64)
        self.block5 = res_block(64)
        self.block6 = res_block(64)
        self.block7 = res_block(64)
        self.block8 = res_block(64)
        self.block9 = res_block(64)
        self.block10 = res_block(64)
        self.block11 = res_block(64)
        self.block12 = res_block(64)
        self.block13 = res_block(64)
        self.block14 = res_block(64)
        self.block15 = res_block(64)
        self.block16 = res_block(64)
        self.block17 = res_block(64)
        self.block18 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))
        upsampling = []
        for out_features in range(2):
            upsampling += [
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.block19 = nn.Sequential(*upsampling)
        self.block20 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.block1(x)
        out = self.block2(out1)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.block10(out)
        out = self.block11(out)
        out = self.block12(out)
        out = self.block13(out)
        out = self.block14(out)
        out = self.block15(out)
        out = self.block16(out)
        out = self.block17(out)
        out2 = self.block18(out)
        out = torch.add(out1, out2)
        out = self.block19(out)
        out = self.block20(out)
        return out