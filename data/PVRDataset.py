from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
import matplotlib.pylab as plt

class PVRDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(
        self, 
        file_path:str,
        transform=None
        ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(PVRDataset, self).__init__()

        self.transform = transform

        file = open(file_path,'rb')    

        x, y = pickle.load(file)

        self.x = torch.from_numpy(np.array(x))

        # if self.x.shape[1] != 3:
        #     self.x = self.x.repeat(1,3,1,1)

        if torch.max(self.x) > 1:
            self.x = self.x / 255

        self.y = torch.from_numpy(np.array(y)).to(torch.int64)

        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        if self.transform:
            return self.transform(self.x[idx]), self.y[idx]

        return self.x[idx], self.y[idx]