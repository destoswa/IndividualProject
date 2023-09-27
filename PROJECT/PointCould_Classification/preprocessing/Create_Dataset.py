import os
import shutil


def MultipleCopies(file_name, folder, num_copies):
    file_path = f"{folder}/{file_name}.pcd"
    if not os.path.isfile(file_path):
        print(f"The file {file_path} can't be found")
        exit()
    for i in range(1, num_copies):
        copy_name = file_name+str(i)+".pcd"
        shutil.copyfile(file_path, folder + "/" + copy_name)
    print("Copies succesfuly done !")


if __name__ == "__main__":
    os.mkdir("data")
    os.mkdir("data/garbadge")
    os.mkdir("data/single")
    os.mkdir("data/multiple")
    shutil.copyfile('example.pcd', './data/garbadge/example_garbadge.pcd')
    MultipleCopies('example_garbadge', './data/garbadge', 100)
    shutil.copyfile('example.pcd', './data/single/example_single.pcd')
    MultipleCopies('example_single', './data/single', 100)
    shutil.copyfile('example.pcd', './data/multiple/example_multiple.pcd')
    MultipleCopies('example_multiple', './data/multiple', 100)

