import os
import pandas as pd
import shutil

from tqdm import tqdm
from src.dataset import ModelTreesDataLoader
from torch.utils.data import DataLoader
from src.utils import *
from models.model import KDE_cls_model
from time import time
import tkinter as tk
from tkinter import messagebox


# ===================================================
# ================= HYPERPARAMETERS =================
# ===================================================
# preprocessing
do_preprocess = True
do_update_caching = True

# inference
batch_size = 12
num_workers = 12
num_class = 3
grid_size = 64
kernel_size = 1
num_repeat_kernel = 2
SRC_INF_ROOT = "./inference/"
SRC_INF_DATA = SRC_INF_ROOT + "data/"
SRC_INF_RESULTS = os.path.join(SRC_INF_ROOT, 'results/')
# SRC_INF_DATA = r"D:\PDM_repo\Github\PDM\data\classification_gt\classification_gt\color_grp_full_tile_128_out_gt_split_instance\data"
SRC_MODEL = "./models/pretrained/model_KDE.tar"
INFERENCE_FILE = "modeltrees_inference.csv"
with open(SRC_INF_ROOT + 'modeltrees_shape_names.txt', 'r') as f:
    SAMPLE_LABELS = f.read().splitlines()

# ===================================================
# ===================================================

# store relation between number and class label
dict_labels = {}
for idx, cls in enumerate(SAMPLE_LABELS):
    dict_labels[idx] = cls


def inference(args):
    print("Loading model...")
    conf = {
        "num_class": args['num_class'],
        "grid_dim": args['grid_size'],
    }

    # load the model
    model = KDE_cls_model(conf).to(torch.device('cuda'))
    checkpoint = torch.load(SRC_MODEL)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # create the folders for results
    if os.path.exists(SRC_INF_RESULTS):
        root = tk.Tk()
        root.withdraw()
        if messagebox.askyesno("Potential data lost!", 'A "results" directory already exists. Do you want to overwrite it?'):
            shutil.rmtree(SRC_INF_RESULTS)
    os.makedirs(SRC_INF_RESULTS)

    # if not os.path.isdir("./inference/results/"):
    #     os.mkdir("./inference/results")
    for cls in SAMPLE_LABELS:
        os.makedirs(os.path.join(SRC_INF_RESULTS, cls), exist_ok=True)

    # preprocess the samples
    if args['do_preprocess']:
        lst_files_to_process = ['data/' + cls for cls in os.listdir(SRC_INF_DATA) if cls.endswith('.pcd')]
        df_files_to_process = pd.DataFrame(lst_files_to_process, columns=['data'])
        df_files_to_process['label'] = 0
        df_files_to_process.to_csv(SRC_INF_ROOT + INFERENCE_FILE, sep=';', index=False)

    # make the predictions
    print("making predictions...")
    kde_transform = ToKDE(args['grid_size'], args['kernel_size'], args['num_repeat_kernel'])
    inferenceSet = ModelTreesDataLoader(INFERENCE_FILE, SRC_INF_ROOT, split='inference', transform=None, do_update_caching=args['do_update_caching'], kde_transform=kde_transform)
    if len(inferenceSet.num_fails) > 0:
        os.makedirs(os.path.join(SRC_INF_RESULTS, 'failures/'), exist_ok=True)
        for _, file_src in inferenceSet.num_fails:
            shutil.copyfile(
                src=file_src, 
                dst=os.path.join(SRC_INF_RESULTS, 'failures/', os.path.basename(file_src)))

    inferenceDataLoader = DataLoader(inferenceSet, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'], pin_memory=True)
    df_predictions = pd.DataFrame(columns=["file_name", "class"])

    for batch_id, data in tqdm(enumerate(inferenceDataLoader, 0), total=len(inferenceDataLoader), smoothing=0.9):
        # load the samples and labels on cuda
        grid, target, filenames = data['grid'], data['label'], data['filename']
        grid, target = grid.cuda(), target.cuda()

        # compute prediction
        pred = model(grid)
        pred_choice = pred.data.max(1)[1]

        # copy samples into right result folder
        for idx, pred in enumerate(pred_choice):
            fn = filenames[idx].replace('.pickle', '')
            dest = "inference/results/" + dict_labels[pred.item()] + "/" + fn.replace('data/', "")
            shutil.copyfile(os.path.abspath('inference/' + fn), os.path.abspath(dest))
            df_predictions.loc[len(df_predictions)] = [fn, pred.item()]

    # save results in csv file
    df_predictions.to_csv(SRC_INF_ROOT + 'results/results.csv', sep=';', index=False)


def main():
    args = {
        'do_preprocess': do_preprocess,
        'do_update_caching': do_update_caching,
        'grid_size': grid_size,
        'num_class': num_class,
        'kernel_size': kernel_size,
        'num_repeat_kernel': num_repeat_kernel,
        'batch_size': batch_size,
        'num_workers': num_workers,
    }
    inference(args)


if __name__ == "__main__":
    start = time()
    main()
    duration = time() - start
    hours = int(duration/3600)
    mins = int((duration - 3600 * hours)/60)
    secs = int((duration - 3600 * hours - 60 * mins))
    print(duration)
    print(f"Time to process inference: {hours}:{mins}:{secs}")

#     pcd_files = [
#         "./inference/data/cluster_11631.pcd",
#         "./inference/data/cluster_113581.pcd",
#         "./inference/data/cluster_127837.pcd",
#         "./inference/data/cluster_102883.pcd",
#         "./inference/data/cluster_122776.pcd",
#         "./inference/data/cluster_107092.pcd",
#         "./inference/data/cluster_127170.pcd",
#         "./inference/data/cluster_121849.pcd",
#         "./inference/data/cluster_109930.pcd",
#         "./inference/data/cluster_10517.pcd",
#         "./inference/data/cluster_11513.pcd",
#         "./inference/data/cluster_10563.pcd",
#         "./inference/data/cluster_117904.pcd",
#         "./inference/data/cluster_107170.pcd",
#         "./inference/data/cluster_110075.pcd",
#         "./inference/data/cluster_117906.pcd",
#         "./inference/data/cluster_111588.pcd",
#         "./inference/data/cluster_114509.pcd",
#         "./inference/data/cluster_107415.pcd",
#         "./inference/data/cluster_120845.pcd",
#         "./inference/data/cluster_124815.pcd",
#         "./inference/data/cluster_11956.pcd",
#         "./inference/data/cluster_123756.pcd",
#         "./inference/data/cluster_111217.pcd",
#         "./inference/data/cluster_114575.pcd",
#         "./inference/data/cluster_109866.pcd",
#     ]
#     for file in pcd_files:
#         os.remove(file)
#     quit()