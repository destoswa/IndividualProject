import numpy as np
import torch
from scipy.stats import multivariate_normal


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        pointCloud, label = sample['pointCloud'], sample['label']

        return {'pointCloud': torch.from_numpy(pointCloud),
                'label': torch.from_numpy(np.asarray(label))}


class ToKDE(object, filter_size=2):
    """Convert pointCloud to KDE vector"""

    def __init__(self, grid_size, filter_size):
        self.grid_size = grid_size
        self.filter_size = filter_size

    def __call__(self, sample):
        pointCloud = sample['pointCloud']
        pointCloud = normalize_data(pointCloud)

        # create KDE grid:
        grid = pcToGrid(pointCloud, self.grid_size, self.filter_size)

        return {'pointCloud': grid,
                'label': sample['label']}


def pcToGrid(data, grid_size, filter_size):
    grid = np.zeros((grid_size, grid_size, grid_size))

    # find position of each point on grid
    for id_point, point in enumerate(data):
        for idx, pos in enumerate(point):
            point[idx] = int((pos + 1)/2*grid_size)
        point = point.astype(int)

        # create KDE grid:
        grid = pcToKDEgrid(grid, point, filter_size)

    return grid


def pcToKDEgrid(grid, point_pos, filter_size):
    x, y, z = np.mgrid[-1:1.1:(1 / filter_size), -1:1.1:(1 / filter_size), -1:1.1:(1 / filter_size)]
    pos = np.stack((x, y, z), axis=-1)
    rv = multivariate_normal([0, 0, 0], .2)
    point_grid = rv.pdf(pos)
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                x_grid = point_pos[0] - filter_size + i
                y_grid = point_pos[1] - filter_size + j
                z_grid = point_pos[2] - filter_size + k
                if grid.shape[0] > x_grid >= 0 and grid.shape[1] > y_grid >= 0 and grid.shape[2] > z_grid >= 0:
                    grid[x_grid, y_grid, z_grid] += point_grid[i, j, k]
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
