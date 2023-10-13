import os
import numpy as np
import pandas as pd
import csv
from src.pp_pointMLP import pp_pointMLP


def preprocess(source_data, frac_train=.8):
    # Create file with different labels:
    with open(f'{source_data}/modeltrees_shape_names.txt', 'w') as f:
        f.write('garbadge\nmultiple\nsingle')

    # Create references files:
    num_training_samples = 0
    num_testing_samples = 0
    df_training_samples = pd.DataFrame(columns=['data', 'label'])
    df_testing_samples = pd.DataFrame(columns=['data', 'label'])
    dic_temp = {}
    for folder in os.listdir(source_data):
        if os.path.isdir(source_data + "/" + folder):
            label = 0
            if folder == 'garbadge':
                label = 0
            elif folder == 'multiple':
                label = 1
            elif folder == 'single':
                label = 2

            # partition lists
            num_files = len(os.listdir(source_data + "/" + folder))
            num_train = int(num_files * frac_train)
            num_training_samples += num_train
            num_testing_samples += num_files - num_train
            list_train = os.listdir(source_data + "/" + folder)[0:num_train]
            list_train_label = [label] * len(list_train)
            list_test = os.listdir(source_data + "/" + folder)[num_train::]
            list_test_label = [label] * len(list_test)
            dic_temp['data'] = list_train
            dic_temp['label'] = list_train_label
            df_temp = pd.DataFrame(dic_temp, columns=['data', 'label'])
            df_training_samples = pd.concat([df_training_samples, df_temp], ignore_index=True)
            dic_temp['data'] = list_test
            dic_temp['label'] = list_test_label
            df_temp = pd.DataFrame(dic_temp, columns=['data', 'label'])
            df_testing_samples = pd.concat([df_testing_samples, df_temp], ignore_index=True)

            """with open(f'{source_data}/modeltrees_train.txt', 'a') as f:
                f.write("\n".join(list_train))
            with open(f'{source_data}/modeltrees_test.txt', 'a') as f:
                f.write("\n".join(list_test))"""
    df_training_samples.to_csv(f'{source_data}/modeltrees_train.csv', ';', index=False)
    df_testing_samples.to_csv(f'{source_data}/modeltrees_test.csv', ';', index=False)
    print(f"Final partition for training : {num_training_samples}")
    print(f"Final partition for testing : {num_testing_samples}")


def main():
    do_pp_pointMLP = False

    if do_pp_pointMLP:
        pp_pointMLP("./data", './data_models/pointMLP', .8)

    preprocess("./data/modeltrees", .8)


if __name__ == "__main__":
    main()
