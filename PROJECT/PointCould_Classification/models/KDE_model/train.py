import numpy as np
import os
import torch
import csv
import pandas as pd
import time
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from dataset import modeltreesDataLoader
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import *
from model import PointTransformerCls
from visualization import show_log_train, show_confusion_matrix

num_class = 3
num_epoch = 10
batch_size = 16
learning_rate = 1e-3
weight_decay = 1e-4

frac_training_data = .01
frac_testing_data = .1
grid_dim = 64

ROOT_DIR = 'data/modeltrees_5200/'
TRAIN_FILES = 'modeltrees_train.csv'
TEST_FILES = 'modeltrees_test.csv'
with open(ROOT_DIR + '/modeltrees_shape_names.txt', 'r') as f:
    SAMPLE_LABELS = f.read().splitlines()


def train_epoch(trainDataLoader, model, optimizer, criterion):
    loss_tot = 0
    num_samp_tot = 0
    mean_correct = []
    model.train()
    for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        grid, target = data['pointCloud'], data['label']
        grid, target = grid.cuda(), target.cuda()
        optimizer.zero_grad()
        pred = model(grid)
        loss = criterion(pred, target.long())
        loss_tot += loss.item()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(grid.size()[0]))
        loss.backward()
        optimizer.step()
        num_samp_tot += grid.shape[0]
    train_acc = np.mean(mean_correct)
    train_loss = loss_tot / num_samp_tot
    return train_acc, train_loss


def test_epoch(testDataLoader, model, criterion):
    loss_tot = 0
    mean_correct = []
    pred_tot = []
    target_tot = []
    class_acc = np.zeros((num_class, 3))
    num_samp_tot = 0
    for batch_id, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
        grid, target = data['pointCloud'], data['label']
        grid, target = grid.cuda(), target.cuda()
        pred = model(grid)
        loss = criterion(pred, target.long())
        loss_tot += loss.item()
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item()/float(grid[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(grid.size()[0]))
        num_samp_tot += grid.size()[0]
        pred_tot.append(pred_choice.tolist())
        target_tot.append(target.tolist())
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    test_acc = np.mean(mean_correct)
    test_loss = loss_tot / num_samp_tot
    pred_tot = [item for sublist in pred_tot for item in sublist]
    target_tot = [item for sublist in target_tot for item in sublist]
    return test_acc, test_loss, class_acc, pred_tot, target_tot


def training(log_version, log_source):
    # check torch and if cuda is available
    print("torch version : " + torch.__version__)
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE")
    else:
        print("Cuda available")

    # transformation
    data_transform = transforms.Compose([ToKDE(grid_dim), ToTensor()])

    # load datasets
    trainingSet = modeltreesDataLoader(TRAIN_FILES, ROOT_DIR, transform=data_transform, frac=frac_training_data)
    testingSet = modeltreesDataLoader(TEST_FILES, ROOT_DIR, transform=data_transform, frac=frac_testing_data)

    trainDataLoader = DataLoader(trainingSet, batch_size=batch_size, shuffle=True, num_workers=4)
    testDataLoader = DataLoader(testingSet, batch_size=batch_size, shuffle=True, num_workers=4)
    torch.manual_seed(2809)

    # get class weights:
    print('Calculating weights...')
    for batch_id, data in enumerate(trainDataLoader, 0):
        targets = data['label'][:] if batch_id == 0 else torch.cat((targets, data['label'][:]), 0)
    targets = targets.numpy()
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(targets),
        y=targets
    )
    print('Weights : ', weights)
    class_weights = torch.tensor(weights, dtype=torch.float, device=torch.device('cuda:0'))

    # model (temporary)
    conf = {
        "num_class": 3,
        "grid_dim": grid_dim,
    }
    model = PointTransformerCls(conf).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)

    # looping on epochs
    best_test_acc = 0
    best_test_class_acc = 0
    best_test_loss = 0
    best_epoch = 0
    for epoch in range(num_epoch):
        line_log = []

        # training
        print(f"Training on epoch {str(epoch+1)}/{str(num_epoch)}:")
        train_acc, train_loss = train_epoch(trainDataLoader, model, optimizer, criterion)
        scheduler.step()
        line_log.append((train_acc, train_loss))
        print("Training acc : ", train_acc)
        print("Training loss : ", train_loss)
        print("Testing...")

        # testing
        with torch.no_grad():
            test_acc, test_loss, class_acc, preds_test, targets_test = test_epoch(testDataLoader, model, criterion)
        line_log.append((test_acc, class_acc, test_loss))
        line_log = [el for sublists in line_log for el in sublists]     # flatten list
        print("Testing acc : ", test_acc)
        print("Testing class acc : ", class_acc)
        print("Testing loss : ", test_loss)
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_test_class_acc = class_acc
            best_test_loss = test_loss

            # save model
            print("Best results : saving model...")
            torch.save(model.state_dict(), log_source + "/model_KDE.pth")

            # save preds and tests for confusion matrix
            conf_mat_data = {
                'pred': preds_test,
                'target': targets_test,
            }
            df_conf_mat_data = pd.DataFrame(conf_mat_data)
            df_conf_mat_data.to_csv('./log/' + str(log_version) + '/confmat.csv', index=False, sep=';')
            show_confusion_matrix(log_source, preds_test, targets_test, SAMPLE_LABELS, epoch=best_epoch)

        # update logs
        with open('./log/' + str(log_version) + '/logs.csv', 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([str(x) for x in line_log])
    print("\n==============\n")
    print("BEST RESULTS ON EPOCH ", best_epoch+1)
    print("BEST TEST ACC: ", best_test_acc)
    print("BEST TEST CLASS ACC: ", best_test_class_acc)
    print("BEST TEST LOSS: ", best_test_loss)


def main():
    # create csv of logs:
    version = 0
    while os.path.exists('./log/' + str(version)):
        version += 1
    os.mkdir('./log/' + str(version))
    log_file_root = './log/' + str(version)
    log_file = './log/' + str(version) + '/logs.csv'
    with open(log_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['train_acc', 'train_loss', 'test_acc', 'test_class_acc', 'test_loss'])

    # Training
    start_time = time.time()
    training(version, log_file_root)
    end_time = time.time()

    # Plots of results
    show_log_train(log_file, log_file_root)

    # print time of training
    delta_time = end_time - start_time
    n_hours = int(delta_time / 3600)
    n_min = int((delta_time % 3600) / 60)
    n_sec = int(delta_time - n_hours * 3600 - n_min * 60)
    print(f"TIME TO TRAIN: {n_hours}:{n_min}:{n_sec}")


if __name__ == "__main__":
    main()