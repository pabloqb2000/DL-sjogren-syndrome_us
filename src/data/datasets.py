import os
import cv2
import torch
from os import path
from torch.utils.data import Dataset
from torchvision.transforms import v2

class CachedImageDataset(Dataset):
    def __init__(self, root_dir, cached_transform=None, online_transform=None):
        self.root_dir = root_dir
        self.online_transform = online_transform
        
        self.images = []
        for folder in os.listdir(root_dir):
            for img in os.listdir(path.join(root_dir, folder)):
                self.images.append(
                    cv2.imread(path.join(root_dir, folder, img))
                )

        if cached_transform:
            self.images = [cached_transform(img) for img in self.images]

        self.labels = torch.Tensor(sum([
            [i]*len(os.listdir(path.join(root_dir, folder)))
            for i, folder in enumerate(os.listdir(root_dir))
        ], [])).to(torch.int64)

        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.online_transform:
            img = self.online_transform(img)
        
        return img, label



class CustomImageDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.images = X
        self.labels = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.images[idx]
        if self.transform:
            x = self.transform(x / 255.0)

        y = self.labels[idx]
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

