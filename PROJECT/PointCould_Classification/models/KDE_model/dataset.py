import os
import shutil
import numpy as np
import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
import open3d as o3d
import concurrent.futures
from functools import partial
from tqdm import tqdm


class ModelTreesDataLoader(Dataset):
    def __init__(self, csvfile, root_dir, split, transform, do_update_caching, kde_transform, frac=1.0):
        """
            Arguments:
                :param csv_file (string): Path to the csv file with annotations
                :param root_dir (string): Directory with the csv files and the folders containing pcd files per class
                :param split (string): type of dataset (train or test)
                :param transform (callable, optional): Optional transform to be applied
                :param frac (float, optional): fraction of the data loaded
                    on a sample.
        """
        # create code for caching grids
        self.transform = transform
        self.root_dir = root_dir
        pickle_dir = root_dir + 'tmp_grids_' + split + "/"
        self.pickle_dir = pickle_dir
        if do_update_caching:
            self.clean_temp()
            os.mkdir(pickle_dir)
            os.mkdir(pickle_dir + "Garbage")
            os.mkdir(pickle_dir + "Multi")
            os.mkdir(pickle_dir + "Single")
        self.data = pd.read_csv(root_dir + csvfile, delimiter=';')
        self.data = self.data.sample(frac=frac, random_state=42).reset_index(drop=True)

        print('Loading ', split, ' set...')
        if do_update_caching:
            # creating grids using multiprocess
            with concurrent.futures.ProcessPoolExecutor() as executor:
                partialmapToKDE = partial(self.mapToKDE, root_dir, pickle_dir, kde_transform)
                args = range(len(self.data))
                results = list(tqdm(executor.map(partialmapToKDE, args), total=len(self.data), smoothing=.9, desc="creating caching files"))

        for idx, samp in tqdm(self.data.iterrows(), total=len(self.data), smoothing=.9, desc="loading file names"):
            self.data.iloc[idx, 0] = samp['data'] + '.pickle'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.data.iloc[idx, 0]

        with open(self.pickle_dir + filename, 'rb') as file:
            sample = pickle.load(file)

        sample = {'grid': sample['data'], 'label': sample['label']}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def clean_temp(self):
        if os.path.exists(self.pickle_dir):
            shutil.rmtree(self.pickle_dir)

    def mapToKDE(self, root_dir, pickle_dir, kde_transform, idx):
        samp = self.data.iloc[idx]
        pcd_name = os.path.join(root_dir, samp['data'])
        pcd = o3d.io.read_point_cloud(pcd_name)
        pointCloud = np.asarray(pcd.points)
        label = np.asarray(samp['label'])
        sample = {'data': pointCloud, 'label': label}
        sample = kde_transform(sample)

        with open(pickle_dir + samp['data'] + '.pickle', 'wb') as file:
            pickle.dump(sample, file)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    ROOT_DIR = 'data/modeltrees_5200/'
    TRAIN_FILES = 'modeltrees_train.csv'
    trainingSet = ModelTreesDataLoader(TRAIN_FILES, ROOT_DIR, split='train', transform=None,
                                       frac=1.0)
    trainDataLoader = DataLoader(trainingSet, batch_size=16, shuffle=True, num_workers=4)
    for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        grid, target = data['grid'], data['label']

