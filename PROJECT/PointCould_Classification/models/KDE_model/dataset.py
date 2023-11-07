import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.utils import shuffle
import open3d as o3d


class modeltreesDataLoader(Dataset):
    def __init__(self, csvfile, root_dir, transform, frac=1.0):
        """
            Arguments:
                :param csv_file (string): Path to the csv file with annotations
                :param root_dir (string): Directory with the csv files and the folders containing pcd files per class
                :param transform (callable, optional): Optional transform to be applied
                :param frac (float, optional): fraction of the data loaded
                    on a sample.
        """
        self.data = pd.read_csv(root_dir + csvfile, delimiter=';')
        self.data = shuffle(self.data, random_state=42)
        self.data = self.data.iloc[:int(frac * len(self.data))]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pcd_name = os.path.join(self.root_dir,
                                self.data.iloc[idx, 0])
        pcd = o3d.io.read_point_cloud(pcd_name)
        pointCloud = np.asarray(pcd.points)
        label = np.asarray(self.data.iloc[idx, 1])
        sample = {'pointCloud': pointCloud, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
