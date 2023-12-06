"""
Author: Benny
Date: Nov 2019
"""
from dataset import ModelNetDataLoader
from visualize_logs import show_log_train, show_confusion_matrix
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
import hydra
import omegaconf
import csv
import time
import pandas as pd
import sklearn



def test(model, loader, num_class=40, batch_size=16):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    loss_tot = 0
    target_tot = []
    pred_tot = []

    for j, data in enumerate(loader):
        targets = data[1][:, 0] if j == 0 else torch.cat((targets, data[1][:, 0]), 0)
    targets = targets.numpy()

    # compute weights
    weights = sklearn.utils.class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(targets),
        y=targets
    )

    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        #classifier = model.eval()
        pred = model(points)

        class_weights = torch.tensor(weights, dtype=torch.float, device=torch.device('cuda:0'))
        criterion_test = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

        #criterion_test = torch.nn.CrossEntropyLoss()
        loss_test = criterion_test(pred, target.long())
        loss_tot += loss_test.item()
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
        pred_tot.append(pred_choice.tolist())
        target_tot.append(target.tolist())

    loss_tot = loss_tot / len(loader) / batch_size
    pred_tot = [item for sublist in pred_tot for item in sublist]
    target_tot = [item for sublist in target_tot for item in sublist]
    class_acc[:,2] = class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc, loss_tot, pred_tot, target_tot


@hydra.main(config_path='config', config_name='cls', version_base='1.2')
def main(args):
    # create csv if not existing:
    version = 0
    while os.path.exists('./log/cls/Menghao/logs_v' + str(version) + '.csv'):
        version += 1
    with open('./log/cls/Menghao/logs_v' + str(version) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['train_acc', 'train_loss', 'test_acc', 'test_class_acc', 'test_loss'])

    print(torch.__version__)
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE")
    else:
        print("Cuda available")
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)
    """print(args)
    print(args.pretty())"""

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    #DATA_PATH = hydra.utils.to_absolute_path('data/modelnet40_normal_resampled/')
    DATA_PATH = hydra.utils.to_absolute_path('data/modeltrees_12000_FIXEDSIZE_1024/')
    with open(DATA_PATH + '\modeltrees_shape_names.txt', 'r') as f:
        SAMPLE_LABELS = f.read().splitlines()
    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train', normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    args.num_class = 3
    args.input_dim = 6 if args.normal else 3
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args).cuda()

    # get class weights:
    for batch_id, data in enumerate(trainDataLoader, 0):
        targets = data[1][:, 0] if batch_id == 0 else torch.cat((targets, data[1][:, 0]), 0)
    targets = targets.numpy()
    weights = sklearn.utils.class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(targets),
        y=targets
    )
    print('Weights : ')
    print(weights)
    class_weights = torch.tensor(weights, dtype=torch.float, device=torch.device('cuda:0'))
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    '''TRAINING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        line_log = []
        loss_tot = 0
        classifier.train()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            pred = classifier(points)
            loss = criterion(pred, target.long())
            loss_tot += loss.item()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        scheduler.step()

        loss_mean = loss_tot / len(trainDataLoader) / args.batch_size
        train_instance_acc = np.mean(mean_correct)
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)

        line_log.append(train_instance_acc)
        line_log.append(loss_mean)

        with torch.no_grad():
            instance_acc, class_acc, loss_test, preds_test, targets_test = test(classifier.eval(), testDataLoader, args.num_class, args.batch_size)

            line_log.append(instance_acc)
            line_log.append(class_acc)
            line_log.append(loss_test)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            logger.info('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = 'best_model.pth'
                logger.info('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

                # save preds and tests for confusion matrix
                conf_mat_data = {
                    'pred': preds_test,
                    'target': targets_test,
                }
                df_conf_mat_data = pd.DataFrame(conf_mat_data)
                df_conf_mat_data.to_csv('./log/cls/Menghao/confmat_v' + str(version) + '.csv', index=False, sep=';')

            global_epoch += 1
        with open('./log/cls/Menghao/logs_v' + str(version) + '.csv', 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([str(x) for x in line_log])
    logger.info('End of training...')
    show_log_train('./log/cls/Menghao/logs_v' + str(version)+'.csv', './log/cls/Menghao', do_show=False, do_save=True)
    show_confusion_matrix('./log/cls/Menghao/confmat_v' + str(version) + '.csv', './log/cls/Menghao/', SAMPLE_LABELS, best_epoch, do_show=False, do_save=True)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()

    # print time of training
    delta_time = end_time - start_time
    n_hours = int(delta_time/3600)
    n_min = int((delta_time % 3600)/60)
    n_sec = int(delta_time - n_hours * 3600 - n_min * 60)
    print(f"TIME TO TRAIN: {n_hours}:{n_min}:{n_sec}")
