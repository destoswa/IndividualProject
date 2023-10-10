import os
import numpy as np
from src.pp_pointMLP import pp_pointMLP

def preprocess(source_data, frac_train=.8):
    # Create file with different labels:
    with open(f'{source_data}/modeltrees_shape_names.txt', 'w') as f:
        f.write('garbadge\nmultiple\nsingle')

    # Create references files:
    num_training_samples = 0
    num_testing_samples = 0
    for folder in os.listdir(source_data):
        if os.path.isdir(source_data + "/" + folder):
            # partition lists
            num_files = len(os.listdir(source_data + "/" + folder))
            num_train = int(num_files * frac_train)
            num_training_samples += num_train
            num_testing_samples += num_files - num_train
            list_train = os.listdir(source_data + "/" + folder)[0:num_train]
            list_test = os.listdir(source_data + "/" + folder)[num_train::]
            with open(f'{source_data}/modeltrees_train.txt', 'a') as f:
                f.write("\n".join(list_train))
            with open(f'{source_data}/modeltrees_test.txt', 'a') as f:
                f.write("\n".join(list_test))

    print(f"Final partition for training : {num_training_samples}")
    print(f"Final partition for testing : {num_testing_samples}")


def main():
    do_pp_pointMLP = False

    if do_pp_pointMLP:
        pp_pointMLP("./data", './data_models/pointMLP', .8)

    preprocess("./data/modeltrees", .8)


if __name__ == "__main__":
    main()
