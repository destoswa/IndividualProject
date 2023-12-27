import numpy as np


def rotate(grid_size=5, degree=2):
    grid = np.zeros((grid_size, grid_size, grid_size))
    for i in range(grid_size):
        grid[i, i, i] = 1

    print(grid[:, :, 0])
    print(grid[:, :, 0].shape)
    print(np.rot90(grid[:, :, 0], 1))
    """for i in range(grid_size):
        grid[:, :, i] = np.rot90(grid[:, :, i], 1)"""
    grid = np.rot90(grid, degree, (0, 1))
    for i in range(grid_size):
        print(grid[:, :, i])


if __name__ == '__main__':
    grid_size = 5
    degree = 2
    rotate(grid_size, degree)
