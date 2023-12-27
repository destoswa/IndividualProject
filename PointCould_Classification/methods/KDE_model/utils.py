import numpy as np
import torch
from scipy.stats import multivariate_normal
import random


class RandRotate(object):
    """ Rotate randomly
        Input:
        GxGxG array: the grid to be updated
    Output:
        GxGxG array: the rotated grid
    """
    def __call__(self, sample):
        num_rot = random.randint(0, 3)
        sample['grid'] = torch.rot90(sample['grid'], num_rot, (0, 1))

        return sample


class RandScale(object):
    """ Randomly scale patches of values in sample
        Input:
        GxGxG array: the grid to be updated
    Output:
        GxGxG array: the scaled grid
    """

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, sample):
        idx_nonzeros = torch.nonzero(sample['grid'])
        num_candidates = torch.count_nonzero(sample['grid'])
        num_of_scales = random.randrange(0, int(num_candidates/self.kernel_size**3))
        scales = [random.uniform(0.5, 1.5) for i in range(num_of_scales)]

        # shuffle indices:
        idx = torch.randperm(idx_nonzeros.shape[0])
        idx_nonzeros = idx_nonzeros[idx, :]
        idx_nonzeros = idx_nonzeros[0:num_of_scales]

        # scales patches
        for id_point, point in enumerate(idx_nonzeros):
            sample['grid'][point[0] - self.kernel_size: point[0] + self.kernel_size,
            point[0] - self.kernel_size: point[0] + self.kernel_size,
            point[0] - self.kernel_size: point[0] + self.kernel_size] *= scales[id_point]

        return sample


class ToKDE(object):
    """ Convert pointCloud to KDE vector """

    def __init__(self, grid_size, kernel_size, num_repeat):
        self.grid_size = grid_size
        self.kernel_size = kernel_size
        self.num_repeat = num_repeat

    def __call__(self, sample):
        pointCloud = sample['data']
        pointCloud = pcNormalize(pointCloud)

        # create grid:
        grid = pcToGrid(pointCloud, self.grid_size)

        # apply KDE:
        for rep in range(self.num_repeat):
            grid = applyKDE(grid, self.grid_size, self.kernel_size, self.num_repeat)

        # to tensor
        grid = torch.from_numpy(grid)
        label = torch.from_numpy(np.asarray(sample['label']))

        return {'data': grid,
                'label': label}


def pcToGrid(data, grid_size):
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

        # add point to grid
        grid[point[0], point[1], point[2]] = 1

        # create KDE grid:
        #grid = applyKDE(grid, point, kernel_size)

    return grid


def applyKDE(grid, grid_size, kernel_size, num_repeat):
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
    rv = multivariate_normal([0, 0, 0])
    point_grid = rv.pdf(pos)*10
    new_grid = np.zeros((grid_size + 2 * kernel_size, grid_size + 2 * kernel_size, grid_size + 2 * kernel_size))
    lst_pos = np.where(grid != 0)

    for ind in range(len(lst_pos[0])):
        coeff = grid[lst_pos[0][ind], lst_pos[1][ind], lst_pos[2][ind]]
        new_grid[lst_pos[0][ind]: lst_pos[0][ind] + 2 * kernel_size + 1,
        lst_pos[1][ind]: lst_pos[1][ind] + 2 * kernel_size + 1,
        lst_pos[2][ind]: lst_pos[2][ind] + 2 * kernel_size + 1,
        ] += point_grid * coeff

    return new_grid[kernel_size:-kernel_size, kernel_size:-kernel_size, kernel_size:-kernel_size]


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


""" ----------------------------------- """
""" --|--- OLD USELESS FUNCTIONS ---|-- """
""" --v-----------------------------v-- """


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
