import os
import numpy as np
import torch
import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class ImageDataLoader(Dataset):
    def __init__(self, root, shape):
        height, _ = shape
        #Create Low Resolution Image by Reducing height and width by a factor of 4
        self.LowResolutionTransform = transforms.Compose(
            [
                transforms.Resize((height // 4, height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
            ]
        )
        #Let original image remain the high resolution image
        self.HighResolutionTransform = transforms.Compose(
            [
                transforms.Resize((height, height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
            ]
        )
        self.files = sorted(glob.glob(root + "/*.*"))


    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.LowResolutionTransform(img)
        img_hr = self.HighResolutionTransform(img)

        return {"lr": img_lr, "hr": img_hr} #return dictionary of images, where "lr" key contains low res images and "hr" key contains high res images

    def __len__(self):
        return len(self.files)