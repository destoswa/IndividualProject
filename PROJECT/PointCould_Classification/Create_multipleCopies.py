import os
import shutil


def MultipleCopies(file_name, folder, num_copies):
    file_path = f"{folder}/{file_name}.pcd"
    if not os.path.isfile(file_path):
        print(f"The file {file_path} can't be found")
        exit()
    for i in range(num_copies):
        copy_name = file_name+str(i)+".pcd"
        shutil.copyfile(file_path, folder + "/" + copy_name)
    print("Copies succesfuly done !")


if __name__ == "__main__":
    """os.chdir("data")
    os.chdir("data/garbadge")
    os.chdir("data/single")
    os.chdir("datamultiple")
    shutil.copyfile('example','example_garbadge')"""
    MultipleCopies('example_garbadge', 'garbadge', 100)

