import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def main():
    grid_size = 20
    cell_size = 1 / grid_size
    num_cells = 2
    point_pos = [4, 4, 4]
    rel_pos = [-1 + point_pos[0] * cell_size, -1 + point_pos[1] * cell_size, -1 + point_pos[2] * cell_size]
    #print('rel_pos : ', rel_pos)

    grid = np.zeros((grid_size, grid_size, grid_size))
    #x, y = np.mgrid[-1:1.1:1/num_cells, -1:1.1:1/num_cells]
    x, y, z = np.mgrid[-1:1.1:1/num_cells, -1:1.1:1/num_cells, -1:1.1:1/num_cells]
    #pos = np.dstack((x, y))
    pos = np.stack((x, y, z), axis=-1)
    rv = multivariate_normal([0, 0, 0], .2)
    point_grid = rv.pdf(pos)

    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                x_grid = point_pos[0] - num_cells + i
                y_grid = point_pos[1] - num_cells + j
                z_grid = point_pos[2] - num_cells + k
                if grid.shape[0] > x_grid >= 0 and grid.shape[1] > y_grid >= 0 and grid.shape[2] > z_grid >= 0:
                    grid[x_grid, y_grid, z_grid] = point_grid[i, j, k]
    print(rv.pdf(pos).shape)
    print(np.round(rv.pdf(pos), 2))
    print(np.round(grid, 2))

    # check if matrix symetric along 3 axis:
    for i in range(3):
        for j in range(3):
            assert grid.all() == np.moveaxis(grid, i, j).all()
    """fig = plt.figure(figsize=(5,5))
    ax = plt.axes(projection='3d')
    #ax = fig2.add_subplot(111)
    linspace = np.linspace(-1, 1, grid_size)
    ax.contour3D(grid)
    plt.show()"""


if __name__ == '__main__':
    main()
