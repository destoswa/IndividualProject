"""
visualizing lidar data
"""

#%%
# Import packages
import numpy as np
import laspy
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# import lidar .las data and assign to variable
las = laspy.read("../data/points.las")

# examine the available featurees for the lidar file we read
print(list(las.point_format.dimension_names))

# exploring data
point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))


tree_crown = np.zeros([1,3])
tree_crown_relative = np.zeros([1,3])
"""print(tree_crown)
print(np.append(tree_crown,[[1,2,3]],axis=0))
exit()"""

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
top_points = 100
top_percent = 0.0
cellsize = max([deltax, deltay])/(resolution-0.0001)

print(f"Density of points per cell = {point_data.shape[0]/resolution**2}")

print(f"min height = {np.min(point_data[:,2])}")
point_data_relative = point_data - np.array([0, 0, np.min(point_data[:,2])])
point_data_grid = {}
for i in range(0, resolution):
    point_data_grid[i] = {}

print("Creating grid...")
for i in tqdm(range(0, resolution)):
    for j in range(0, resolution):
        point_data_grid[i][j] = pd.DataFrame(columns=['X', 'Y', 'Z'])

print("Assigning points in grid...")
for point in tqdm(point_data):
    xpos = np.floor((point[0] - xmin) / cellsize)
    ypos = np.floor((point[1] - ymin) / cellsize)
    point_data_grid[xpos][ypos] = pd.concat([point_data_grid[xpos][ypos], pd.DataFrame(point.reshape(1, -1), columns=['X', 'Y', 'Z'])], ignore_index=True)

df_tree_crown = pd.DataFrame(columns=['X', 'Y', 'Z'])
df_tree_crown_relative = pd.DataFrame(columns=['X', 'Y', 'Z'])
"""print("Sorting points in each cell...")
for i in tqdm(range(0, resolution)):
    for j in range(0, resolution):
        point_data_grid[i][j] = point_data_grid[i][j].sort_values(by=['Z'])

        # Taking the top points in cell
        if len(point_data_grid[i][j]) > 0:
            min_height_in_cell = point_data_grid[i][j].iloc[0].copy()
            min_height_in_cell['X'] = 0
            min_height_in_cell['Y'] = 0
        if len(point_data_grid[i][j]) > top_points:
            df_tree_crown = pd.concat([df_tree_crown, point_data_grid[i][j][-top_points:]])
            df_tree_crown_relative = pd.concat([df_tree_crown_relative, point_data_grid[i][j][-top_points:] - min_height_in_cell])
        elif len(point_data_grid[i][j]) > 0:
            df_tree_crown = pd.concat([df_tree_crown, point_data_grid[i][j]])
            df_tree_crown_relative = pd.concat([df_tree_crown_relative, point_data_grid[i][j][:] - min_height_in_cell])"""
print("Sorting points in each cell...")

for i in tqdm(range(0, resolution)):
    for j in range(0, resolution):
        point_data_grid[i][j] = point_data_grid[i][j].sort_values(by=['Z'])

        # Taking the top points in cell
        if len(point_data_grid[i][j]) > 0:
            min_height_in_cell = point_data_grid[i][j].iloc[0].copy()
            min_height_in_cell['X'] = 0
            min_height_in_cell['Y'] = 0
            max_height_in_cell = point_data_grid[i][j].iloc[-1].copy()
            deltaz = max_height_in_cell - min_height_in_cell
            treshold_height_in_cell = min_height_in_cell['Z'] + top_percent * deltaz['Z']
            for index, point in point_data_grid[i][j].iterrows():
                if point['Z'] > treshold_height_in_cell:
                    tree_crown = np.append(tree_crown, [point.to_numpy()], axis=0)
                    tree_crown_relative = np.append(tree_crown_relative, [(point - min_height_in_cell.squeeze()).to_numpy()], axis=0)
                    """df_tree_crown = pd.concat([df_tree_crown, point], axis=1).T
                    df_tree_crown_relative = pd.concat([df_tree_crown_relative, point - min_height_in_cell], axis=1, ignore_index=True)
                    df_tree_crown = pd.DataFrame([df_tree_crown, point])
                    df_tree_crown_relative = pd.DataFrame([df_tree_crown_relative, point - min_height_in_cell])"""

"""print(df_tree_crown_relative)
print(df_tree_crown)
print(point_data)
print(point_data_relative)
tree_crown = df_tree_crown.to_numpy()
tree_crown_relative = df_tree_crown_relative.to_numpy()"""
print("tree crown:")
tree_crown = np.delete(tree_crown, 0, axis=0)
tree_crown_relative = np.delete(tree_crown_relative, 0, axis=0)
print(tree_crown)
print(f"tree_crown_relative : \n{tree_crown_relative}")
# 3D point cloud visualization

#Visualization 1
geom = o3d.geometry.PointCloud()
#geom.points = o3d.utility.Vector3dVector(tree_crown)
#geom.points = o3d.utility.Vector3dVector(point_data_relative)
geom.points = o3d.utility.Vector3dVector(tree_crown_relative)
#geom.points = o3d.utility.Vector3dVector(point_data)
o3d.visualization.draw_geometries([geom])
"""image = o3d.visualization.Visualizer()
image.capture_screen_image("../figures/point_cloud.png")"""

"""#Visualization 2
geom = o3d.geometry.PointCloud()
geom.points = o3d.utility.Vector3dVector(tree_crown_relative)
o3d.visualization.draw_geometries([geom])"""

# tiff DEM visualization
"""img = plt.imread('../data/points.tif')
plt.imshow(img)
plt.show()"""
