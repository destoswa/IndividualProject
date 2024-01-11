import os
import pandas as pd
import shutil

from tqdm import tqdm
from src.dataset import ModelTreesDataLoader
from torch.utils.data import DataLoader
from src.utils import *
from models.model import KDE_cls_model

do_preprocess = True
do_update_caching = True
batch_size = 12
num_workers = 12
num_class = 3
grid_size = 64
kernel_size = 1
num_repeat_kernel = 2
SRC_INF_ROOT = "./inference/"
SRC_INF_DATA = SRC_INF_ROOT + "data/"
SRC_MODEL = "./models/pretrained/model_KDE.tar"
INFERENCE_FILE = "modeltrees_inference.csv"

with open(SRC_INF_ROOT + 'modeltrees_shape_names.txt', 'r') as f:
    SAMPLE_LABELS = f.read().splitlines()

dict_labels = {}
for idx, cls in enumerate(SAMPLE_LABELS):
    dict_labels[idx] = cls


def inference(args):
    print("Loading model...")
    conf = {
        "num_class": args['num_class'],
        "grid_dim": args['grid_size'],
    }
    model = KDE_cls_model(conf).to(torch.device('cuda'))
    checkpoint = torch.load(SRC_MODEL)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Creating folders...")
    for cls in SAMPLE_LABELS:
        if not os.path.isdir("./inference/results/" + cls):
            os.mkdir("./inference/results/" + cls)

    if args['do_preprocess']:
        lst_files_to_process = ['data/' + cls for cls in os.listdir(SRC_INF_DATA)]
        df_files_to_process = pd.DataFrame(lst_files_to_process, columns=['data'])
        df_files_to_process['label'] = 0
        df_files_to_process.to_csv(SRC_INF_ROOT + INFERENCE_FILE, sep=';', index=False)

    print("making predictions...")
    kde_transform = ToKDE(args['grid_size'], args['kernel_size'], args['num_repeat_kernel'])
    inferenceSet = ModelTreesDataLoader(INFERENCE_FILE, SRC_INF_ROOT, split='inference', transform=None, do_update_caching=args['do_update_caching'], kde_transform=kde_transform)
    inferenceDataLoader = DataLoader(inferenceSet, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'], pin_memory=True)
    df_predictions = pd.DataFrame(columns=["file_name", "class"])

    for batch_id, data in tqdm(enumerate(inferenceDataLoader, 0), total=len(inferenceDataLoader), smoothing=0.9):
        grid, target, filenames = data['grid'], data['label'], data['filename']
        grid, target = grid.cuda(), target.cuda()
        pred = model(grid)
        pred_choice = pred.data.max(1)[1]
        for idx, pred in enumerate(pred_choice):
            fn = filenames[idx].replace('.pickle', '')
            dest = "inference/results/" + dict_labels[pred.item()] + "/" + fn.replace('data/', "")
            shutil.copyfile(os.path.abspath('inference/' + fn), os.path.abspath(dest))
            df_predictions.loc[len(df_predictions)] = [fn, pred.item()]

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
    main()
