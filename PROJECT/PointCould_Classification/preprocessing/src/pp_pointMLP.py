import os
import numpy as np
import open3d as o3d
import h5py


def pp_pointMLP(source_data, destination_data, frac_train=.8):
    # initiate partitions
    partition_train = np.zeros((1, 2048, 3))
    partition_test = np.zeros((1, 2048, 3))
    for folder in os.listdir(source_data):
        if os.path.isdir(source_data + "/" + folder):
            # partition lists
            num_files = len(os.listdir(source_data + "/" + folder))
            num_train = int(num_files * frac_train)
            list_train = os.listdir(source_data + "/" + folder)[0:num_train]
            list_test = os.listdir(source_data + "/" + folder)[num_train::]

            # add data to partitions
            for f in list_train:
                pcd = o3d.io.read_point_cloud(source_data + '/' + folder + "/" + f)
                point_cloud = np.asarray(pcd.points)
                point_cloud = np.concatenate((point_cloud, np.zeros((2048-point_cloud.shape[0], 3))))
                partition_train = np.concatenate((partition_train, point_cloud.reshape((1, 2048, 3))))
            for f in list_test:
                pcd = o3d.io.read_point_cloud(source_data + '/' + folder + "/" + f)
                point_cloud = np.asarray(pcd.points)
                point_cloud = np.concatenate((point_cloud, np.zeros((2048-point_cloud.shape[0], 3))))
                partition_test = np.concatenate((partition_test, point_cloud.reshape((1, 2048, 3))))
    partition_train = np.delete(partition_train, 0, axis=0)
    partition_test = np.delete(partition_test, 0, axis=0)
    print("Final partition for training : ", partition_train.shape)
    print("Final partition for testing : ", partition_test.shape)

    # create h5 files in destination folder
    h5f_train = h5py.File(f'{destination_data}/ply_data_train.h5', 'w')
    h5f_train.create_dataset('data', data=partition_train)
    h5f_train.close()
    h5f_test = h5py.File(f'{destination_data}/ply_data_test.h5', 'w')
    h5f_test.create_dataset('data', data=partition_test)
    h5f_test.close()


if __name__ == '__main__':
    pp_pointMLP('../data', '../data_models/pointMLP')
