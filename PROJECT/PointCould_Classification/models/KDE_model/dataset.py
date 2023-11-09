import os
import shutil
import numpy as np
import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
from sklearn.utils import shuffle
import open3d as o3d
from tqdm import tqdm


class ModelTreesDataLoader(Dataset):
    def __init__(self, csvfile, root_dir, split, transform, frac=1.0):
        """
            Arguments:
                :param csv_file (string): Path to the csv file with annotations
                :param root_dir (string): Directory with the csv files and the folders containing pcd files per class
                :param split (string): type of dataset (train or test)
                :param transform (callable, optional): Optional transform to be applied
                :param frac (float, optional): fraction of the data loaded
                    on a sample.
        """
        do_test = True
        # create code for caching grids
        self.root_dir = root_dir
        pickle_dir = root_dir + 'tmp_grids_' + split + "/"
        self.pickle_dir = pickle_dir
        if not do_test:
            os.mkdir(pickle_dir)
            os.mkdir(pickle_dir + "Garbage")
            os.mkdir(pickle_dir + "Multi")
            os.mkdir(pickle_dir + "Single")
        self.data = pd.read_csv(root_dir + csvfile, delimiter=';')
        self.data = shuffle(self.data, random_state=42)
        #self.data = self.data.iloc[:int(frac * len(self.data))]
        self.data = self.data.sample(frac=frac).reset_index(drop=True)
        print('Loading ', split, ' set...')
        for idx, samp in tqdm(self.data.iterrows(), total=len(self.data), smoothing=.9):
            if not do_test:
                pcd_name = os.path.join(root_dir, samp['data'])
                pcd = o3d.io.read_point_cloud(pcd_name)
                pointCloud = np.asarray(pcd.points)
                label = np.asarray(samp['label'])
                sample = {'data': pointCloud, 'label': label}
                if transform:
                    sample = transform(sample)

                with open(pickle_dir + samp['data'] + '.pickle', 'wb') as file:
                    pickle.dump(sample, file)

            self.data.iloc[idx, 0] = samp['data'] + '.pickle'
        #self.data = pd.read_csv(root_dir + csvfile, delimiter=';')
        #self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.data.iloc[idx, 0]
        label = np.asarray(self.data.iloc[idx, 1])

        with open(self.pickle_dir + filename, 'rb') as file:
            sample = pickle.load(file)

        sample = {'grid': sample['data'], 'label': sample['label']}
        """pcd_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        pcd = o3d.io.read_point_cloud(pcd_name)
        pointCloud = np.asarray(pcd.points)
        label = np.asarray(self.data.iloc[idx, 1])
        sample = {'pointCloud': pointCloud, 'label': label}

        if self.transform:
            sample = self.transform(sample)"""

        return sample

    def clean_temp(self):
        shutil.rmtree(self.pickle_dir)

from torch.utils.data import DataLoader
if __name__ == '__main__':
    ROOT_DIR = 'data/modeltrees_5200/'
    TRAIN_FILES = 'modeltrees_train.csv'
    trainingSet = ModelTreesDataLoader(TRAIN_FILES, ROOT_DIR, split='train', transform=None,
                                       frac=1.0)
    trainDataLoader = DataLoader(trainingSet, batch_size=16, shuffle=True, num_workers=4)
    for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        grid, target = data['grid'], data['label']

