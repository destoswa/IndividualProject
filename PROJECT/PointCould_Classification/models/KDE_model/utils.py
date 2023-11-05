import numpy as np
import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        pointCloud, label = sample['pointCloud'], sample['label']

        return {'pointCloud': torch.from_numpy(pointCloud),
                'label': torch.from_numpy(np.asarray(label))}


class ToKDE(object):
    """Convert pointCloud to KDE vector"""

    def __init__(self, grid_size):
            self.grid_size = grid_size

    def __call__(self, sample):
        pointCloud = sample['pointCloud']
        pointCloud = normalize_data(pointCloud)

        # create grid:
        grid = pcToGrid(pointCloud, self.grid_size)

        return {'pointCloud': grid,
                'label': sample['label']}


def pcToGrid(data, grid_size):
    grid = np.zeros((grid_size, grid_size, grid_size))
    for id_point, point in enumerate(data):
        for idx, pos in enumerate(point):
            point[idx] = int((pos + 1)/2*grid_size)
        data[id_point, :] = point
        point = point.astype(int)
        grid[point[0], point[1], point[2]] = 1
    return grid


def normalize_data(data):
    """ Normalize the data, use coordinates of the block centered at origin,
        Input:
            NxC array
        Output:
            NxC array
    """
    pc = data
    centroid = np.mean(data, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    normal_data = pc
    return normal_data
