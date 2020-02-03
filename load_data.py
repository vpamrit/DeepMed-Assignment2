import os
import torch
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pandas import DataFrame, read_csv
from skimage import io, transform #might be the cause of torchvision failures


class SkinData(Dataset):

    def __init__(self, labels_file, root_dir, transform=None):
        self.labels = pd.read_csv(labels_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0];

    def get_data(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target_row = self.labels.iloc[idx, :]
        target = target_row.iloc[1:].to_numpy()

        img_name = target_row.iloc[0]
        image = io.imread(self.root_dir + img_name + ".jpg")

        if self.transform:
            image = self.transform(image)

        return img_name, image, target

    def __getitem__(self, idx):
        img_name, raw_image, label = self.get_data(idx)

        target = torch.from_numpy(label.astype('float').reshape(-1, 7))
        image = torch.from_numpy(image.transpose((2,0,1)))

        return image, target


