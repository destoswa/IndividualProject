import os
import shutil


def MultipleCopies(file_name, folder, num_copies):
    file_path = f"{folder}/{file_name}.pcd"
    if not os.path.isfile(file_path):
        print(f"The file {file_path} can't be found")
        exit()
    for i in range(1, num_copies):
        copy_name = file_name+"_"+str(i)+".pcd"
        shutil.copyfile(file_path, folder + "/" + copy_name)
    os.remove(folder + "/" + file_name + ".pcd")
    print("Copies succesfuly done !")


if __name__ == "__main__":
    root = "data/modeltrees"
    os.mkdir(root)
    os.mkdir(f"{root}/garbadge")
    os.mkdir(f"{root}/single")
    os.mkdir(f"{root}/multiple")
    shutil.copyfile('example1.pcd', f'{root}/garbadge/garbadge.pcd')
    MultipleCopies('garbadge', f'{root}/garbadge', 100)
    shutil.copyfile('example2.pcd', f'{root}/single/single.pcd')
    MultipleCopies('single', f'{root}/single', 300)
    shutil.copyfile('example3.pcd', f'{root}/multiple/multiple.pcd')
    MultipleCopies('multiple', f'{root}/multiple', 250)

