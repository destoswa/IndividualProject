import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from time import time


def func1(grid, grid_size, num_cells, point_grid):
    new_grid = np.zeros((grid_size, grid_size, grid_size))
    lst_pos = np.where(grid != 0)
    for ind in range(len(lst_pos[0])):
        for i in range(2*num_cells + 1):
            for j in range(2*num_cells + 1):
                for k in range(2*num_cells + 1):
                    """x_grid = point_pos[0] - num_cells + i
                    y_grid = point_pos[1] - num_cells + j
                    z_grid = point_pos[2] - num_cells + k"""
                    x_grid = lst_pos[0][ind] - num_cells + i
                    y_grid = lst_pos[1][ind] - num_cells + j
                    z_grid = lst_pos[2][ind] - num_cells + k
                    if grid.shape[0] > x_grid >= 0 and grid.shape[1] > y_grid >= 0 and grid.shape[2] > z_grid >= 0:
                        coeff = grid[lst_pos[0][ind], lst_pos[1][ind], lst_pos[2][ind]]
                        new_grid[x_grid, y_grid, z_grid] += coeff * point_grid[i, j, k]
    return new_grid


"""def func2(grid, grid_size, num_cells, point_grid):
    new_grid = np.zeros((grid_size, grid_size, grid_size))
    lst_pos = np.where(grid != 0)
    for ind in range(len(lst_pos[0])):
        mask = np.zeros((grid_size + 2 * num_cells, grid_size + 2 * num_cells, grid_size + 2 * num_cells))
        coeff = grid[lst_pos[0][ind], lst_pos[1][ind], lst_pos[2][ind]]

        mask[lst_pos[0][ind]: lst_pos[0][ind] + 2 * num_cells + 1,
                 lst_pos[1][ind]: lst_pos[1][ind] + 2 * num_cells + 1,
                 lst_pos[2][ind]: lst_pos[2][ind] + 2 * num_cells + 1,
                 ] = point_grid * coeff
        new_grid += mask[num_cells:-num_cells, num_cells:-num_cells, num_cells:-num_cells]
    return new_grid"""


def func2(grid, grid_size, num_cells, point_grid):
    new_grid = np.zeros((grid_size + 2 * num_cells, grid_size + 2 * num_cells, grid_size + 2 * num_cells))
    lst_pos = np.where(grid != 0)
    for ind in range(len(lst_pos[0])):
        coeff = grid[lst_pos[0][ind], lst_pos[1][ind], lst_pos[2][ind]]

        new_grid[lst_pos[0][ind]: lst_pos[0][ind] + 2 * num_cells + 1,
                 lst_pos[1][ind]: lst_pos[1][ind] + 2 * num_cells + 1,
                 lst_pos[2][ind]: lst_pos[2][ind] + 2 * num_cells + 1,
                 ] += point_grid * coeff
    return new_grid[num_cells:-num_cells, num_cells:-num_cells, num_cells:-num_cells]


def main():
    grid_size = 50
    cell_size = 1 / grid_size
    num_cells = 2
    num_repeat = 2
    point_pos = [2, 2, 2]
    rel_pos = [-1 + point_pos[0] * cell_size, -1 + point_pos[1] * cell_size, -1 + point_pos[2] * cell_size]
    #print('rel_pos : ', rel_pos)

    grid = np.zeros((grid_size, grid_size, grid_size))
    grid[25, 25, 25] = 1
    #grid[point_pos[0], point_pos[1], point_pos[2]] = 1
    #grid[2, 2, 4] = 1
    #x, y = np.mgrid[-1:1.1:1/num_cells, -1:1.1:1/num_cells]
    x, y, z = np.mgrid[-1:1.1:1/num_cells, -1:1.1:1/num_cells, -1:1.1:1/num_cells]
    #pos = np.dstack((x, y))
    pos = np.stack((x, y, z), axis=-1)
    rv = multivariate_normal([0, 0, 0])
    point_grid = rv.pdf(pos)*10
    # test func 1 perf
    start = time()
    for rep in range(num_repeat):
        grid = func1(grid, grid_size, num_cells, point_grid)
    duration_func1 = time() - start
    print(f"Duration of func1: {str(duration_func1)}")

    grid = np.zeros((grid_size, grid_size, grid_size))
    grid[49, 0, 0] = 1
    # test func 2 perf
    start = time()
    for rep in range(num_repeat):
        grid = func2(grid, grid_size, num_cells, point_grid)
    duration_func2 = time() - start
    print(f"Duration of func2: {str(duration_func2)}")
    print(f"Func2 is {str(round(duration_func1/duration_func2, 2))} times faster")
    """print(rv.pdf(pos).shape)
    print(np.round(rv.pdf(pos), 2))
    print(np.round(grid, 2))"""

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
    """a = np.array([[[0, 2], [2, 2]],
                  [[0, 2], [2, 8]]])
    print(a)
    print(a.shape)
    lst_pos = np.where(a == 2)
    print('All index value of 2 is :', lst_pos)
    for ind in range(len(lst_pos[0])):
        #print(pos)
        print(str(lst_pos[0][ind]) + ',' + str(lst_pos[1][ind]) + ',' + str(lst_pos[2][ind]))"""
