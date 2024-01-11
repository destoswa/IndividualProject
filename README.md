# IndividualProject
This project tests different methods to apply classification on 3D point clouds. The dataset on which those models are trained consists in samples from a LIDAR measurement that should represent single trees. However, the algorithm used to segment out those samples shows poor accuracy. Hence, the task at hand is to determine if each sample is one of the following:
- Single tree
- Multiple trees
- Garbage

In the folder _PointCloud_Classification_, 3 methods are either implemented or taken from other repositories and adapted to the project. Those methods are following:
- the 3DmFV-Net method which has been imported from the following repo: https://github.com/sitzikbs/3DmFV-Net
- the pointTransformer method which has been imported from the following repo: https://github.com/qq456cvb/Point-Transformers
- the KDE method which has been implemented from scratch.

Each of those methods contain there own README.md file