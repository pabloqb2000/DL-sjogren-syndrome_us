import os
import cv2
import torch
from os import path
from torch.utils.data import Dataset

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
