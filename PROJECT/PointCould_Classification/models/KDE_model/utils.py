import numpy as np
import torch
from scipy.stats import multivariate_normal


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        return {'data': torch.from_numpy(data),
                'label': torch.from_numpy(np.asarray(label))}


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
        grid = gridNormalize(grid, 'minmax')

        return {'data': grid,
                'label': sample['label']}


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
        grid = pcToKDEgrid(grid, point, kernel_size)

    return grid


def pcToKDEgrid(grid, point_pos, kernel_size):
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
