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
pcd = o3d.io.read_point_cloud("../data/group000383.pcd")
point_cloud = np.asarray(pcd.points)
geom = o3d.geometry.PointCloud()
geom.points = o3d.utility.Vector3dVector(point_cloud)
o3d.visualization.draw_geometries([geom])
exit()

print(list(las.point_format.dimension_names))


point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))

point_label = np.stack(las.classification, axis=0)

print(point_data)
print(point_label)
