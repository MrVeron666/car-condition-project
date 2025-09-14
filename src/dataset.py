import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A      
from albumentations.pytorch import ToTensorV2

class CarConditionDataset(Dataset):
    def __init__(self, df, images_dir, transforms=None): 
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = os.path.join(self.images_dir, row['image'])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        labels = np.array([row['clean'], row['damaged']], dtype=np.float32)
        return image, torch.from_numpy(labels)
