import os
import torch
import csv
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

class OnionDataset(Dataset):
    def __init__(self, csv_file, data_dir, root_dir):

        self.csv_filename = os.path.join(root_dir, csv_file) 
        self.data_dir = os.path.join(root_dir, data_dir)
        self.csv_file = pd.read_csv(self.csv_filename)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imagename_i = os.path.join(self.data_dir, \
                                   self.csv_file.iloc[idx, 0])
        imagename_f = os.path.join(self.data_dir, \
                                   self.csv_file.iloc[idx, 5])

        image_i = transforms.functional.to_tensor(
                      transforms.functional.crop(
                          transforms.functional.to_grayscale(Image.open(imagename_i, 'r'), 1),\
                          0,0,100,124))
        image_f = transforms.functional.to_tensor(
                      transforms.functional.crop(
                          transforms.functional.to_grayscale(Image.open(imagename_f, 'r'), 1),\
                          0,0,100,124))


        u = self.csv_file.iloc[idx,1:5].to_numpy(dtype=np.double)

        sample = {'image_i': image_i, 'image_f': image_f, 'u': u}

        return sample

                                
    

