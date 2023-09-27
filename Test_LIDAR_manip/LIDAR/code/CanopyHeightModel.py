"""
visualizing CanopyHeightModel
"""

# Import packages
import numpy as np
import laspy
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def plotCHM(Y_span, X_span, CHM, filename):
    # Data extraction:
    """X = dataSet['num_retrain'].unique()
    Y = dataSet['top_retrain'].unique()
    Z = np.zeros((len(Y), len(X)))"""
    X = np.arange(X_span)
    Y = np.arange(Y_span)
    Z = CHM
    """for i, x in enumerate(X):
        for j, y in enumerate(Y):
            val = dataSet.loc[(dataSet['num_retrain'] == x) & (dataSet['top_retrain'] == y)]
            Z[j, i] = val['RMSE']"""

    # Plotting figure:
    plt.rcParams.update({'font.size': 14})

    fig = plt.figure()
    ax = plt.subplot()
    c = ax.pcolor(X, Y, Z, cmap='coolwarm')
    """ax.set_title(title)
    ax.set_xlabel('# training iterations [-]')
    ax.set_ylabel('# added elements [-]')
    ax.set_xticks(np.arange(int((X[-1] + 1) / 5), X[-1] + 1, int((X[-1] + 1) / 5)))
    ax.set_yticks(np.arange(int((Y[-1] + 1) / 5), Y[-1] + 1, int((Y[-1] + 1) / 5)))
    cbar.set_label('RMSE [-]')"""
    cbar = fig.colorbar(c, ax=ax)
    plt.tight_layout()
    plt.savefig(f"../figures/{filename}.png")
    plt.draw()


# import lidar .las data and assign to variable
las = laspy.read("../data/points.las")

# examine the available featurees for the lidar file we read
print(list(las.point_format.dimension_names))

# exploring data
point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))

# 3D points manipulation
xmin = np.min(point_data[:, 0])
xmax = np.max(point_data[:, 0])
ymin = np.min(point_data[:, 1])
ymax = np.max(point_data[:, 1])
zmin = np.min(point_data[:, 2])
zmax = np.max(point_data[:, 2])
deltax = xmax - xmin
deltay = ymax - ymin
resolution = 100
xcellsize = deltax/(resolution)
ycellsize = deltay/(resolution)
cellsize = max([deltax, deltay])/(resolution)

print(f"Density of points per cell = {point_data.shape[0]/resolution**2}")

print(f"min height = {np.min(point_data[:,2])}")
point_data_relative = point_data - np.array([0, 0, np.min(point_data[:,2])])
#point_data_grid = np.zeros((resolution,resolution))
point_data_grid = np.zeros((int(deltax/xcellsize), int(deltay/ycellsize)))
point_data_grid_min = np.zeros((int(deltax/xcellsize), int(deltay/ycellsize)))
point_data_grid_relative = np.zeros((int(deltax/xcellsize), int(deltay/ycellsize)))
print(point_data_grid.shape)
print("Assigning points in grid...")
for point in tqdm(point_data):
    xpos = int(np.floor((point[0] - xmin) / xcellsize))
    ypos = int(np.floor((point[1] - ymin) / ycellsize))
    if xpos < point_data_grid.shape[0] and ypos < point_data_grid.shape[1]:
        if point_data_grid_min[xpos][ypos] == 0 or point[2] < point_data_grid_min[xpos][ypos]:
            point_data_grid_min[xpos][ypos] = point[2]
        if point[2] > point_data_grid[xpos][ypos]:
            point_data_grid[xpos][ypos] = point[2]

for i in range(point_data_grid.shape[0]):
    for j in range(point_data_grid.shape[1]):
        point_data_grid_relative[i][j] = point_data_grid[i][j] - point_data_grid_min[i][j]
"""for cell in point_data_grid:
    print(cell)
    if cell == 0:
        cell = zmin"""
plotCHM(point_data_grid.shape[0], point_data_grid.shape[1], point_data_grid, 'HeightModel')
plotCHM(point_data_grid_relative.shape[0], point_data_grid_relative.shape[1], point_data_grid_relative, 'HeightModel_relative')

plt.show()