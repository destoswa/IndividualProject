import numpy as np
import torch
from scipy.stats import multivariate_normal
import random


class RandRotate(object):
    """Rotate randomly"""
    def __call__(self, sample):
        num_rot = random.randint(0, 3)
        sample['grid'] = torch.rot90(sample['grid'], num_rot, (0, 1))

        return sample


class RandScale(object):
    """ randomly scale patches of values in sample """

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, sample):
        idx_nonzeros = torch.nonzero(sample['grid'])
        num_candidates = torch.count_nonzero(sample['grid'])
        num_of_scales = random.randrange(0, int(num_candidates/self.kernel_size**3))
        scales = [random.uniform(0.5, 1.5) for i in range(num_of_scales)]
        #shuffle indices:
        idx = torch.randperm(idx_nonzeros.shape[0])
        idx_nonzeros = idx_nonzeros[idx, :]
        #idx_nonzeros = idx_nonzeros.view(-1, 3)[idx, :].view(idx_nonzeros.size(), 3)

        idx_nonzeros = idx_nonzeros[0:num_of_scales]
         # scales patches
        for id_point, point in enumerate(idx_nonzeros):
            sample['grid'][point[0] - self.kernel_size: point[0] + self.kernel_size,
            point[0] - self.kernel_size: point[0] + self.kernel_size,
            point[0] - self.kernel_size: point[0] + self.kernel_size] *= scales[id_point]
        #grid[idx_nonzeros] = grid[idx_nonzeros] * scales

        return sample


class ToKDE(object):
    """Convert pointCloud to KDE vector"""

    def __init__(self, grid_size, kernel_size):
        self.grid_size = grid_size
        self.kernel_size = kernel_size

    def __call__(self, sample):
        pointCloud = sample['data']
        pointCloud = pcNormalize(pointCloud)

        # create KDE grid:
        grid = pcToGrid(pointCloud, self.grid_size, self.kernel_size)

        # to tensor
        grid = torch.from_numpy(grid)
        label = torch.from_numpy(np.asarray(sample['label']))

        return {'data': grid,
                'label': label}


def pcToGrid(data, grid_size, kernel_size):
    """ Create a grid with a KDE with respect to the point cloud
        Input:
            GxGxG array: the grid to be updated
            3x1 array: the position of the point
            int : the size of the kernel
        Output:
            NxC array
    """
    grid = np.zeros((grid_size, grid_size, grid_size))

    # find position of each point on grid
    for id_point, point in enumerate(data):
        for idx, pos in enumerate(point):
            point[idx] = int((pos + 1)/2*grid_size)
        point = point.astype(int)

        # create KDE grid:
        grid = gridToKDEgrid(grid, point, kernel_size)

    return grid


def gridToKDEgrid(grid, point_pos, kernel_size):
    """ Create a grid with a KDE with respect to one point
        Input:
            GxGxG array: the grid to be updated
            3x1 array: the position of the point
            int : the size of the kernel
        Output:
            NxC array
    """
    x, y, z = np.mgrid[-1:1.1:(1 / kernel_size), -1:1.1:(1 / kernel_size), -1:1.1:(1 / kernel_size)]
    pos = np.stack((x, y, z), axis=-1)
    rv = multivariate_normal([0, 0, 0], .2)
    point_grid = rv.pdf(pos)
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                x_grid = point_pos[0] - kernel_size + i
                y_grid = point_pos[1] - kernel_size + j
                z_grid = point_pos[2] - kernel_size + k
                if grid.shape[0] > x_grid >= 0 and grid.shape[1] > y_grid >= 0 and grid.shape[2] > z_grid >= 0:
                    grid[x_grid, y_grid, z_grid] += point_grid[i, j, k]
    return grid


""" ----------------------------------- """
""" ------ OLD USELESS FUNCTIONS ------ """
""" ----------------------------------- """


def pcNormalize(data):
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


def gridNormalize(grid, method='minmax'):
    if method == 'minmax':
        minVal = np.min(grid)
        maxVal = np.max(grid)
        grid = (grid - minVal)/(maxVal - minVal)
    elif method == 'white':
        raise NotImplementedError('This method still need to be implemented')
    else:
        raise ValueError('The parameter for the argument "method" is wrong.')
    return grid


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        return {'data': torch.from_numpy(data),
                'label': torch.from_numpy(np.asarray(label))}
