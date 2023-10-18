import argparse

from matplotlib import pyplot as plt
import numpy as np
import pickle
import sklearn
import matplotlib.cm as cm
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='./log/modelnet3/3dmfv_net_cls/grid5_log_trial/45', help='Path from root of the project to the file \'log_train.txt\'')


def line_to_float(line):
    print(line)
    return float(line.split(':')[1].replace(' ', ''))


def show_log_train(src, do_save=True, do_show=False):
    ls_train_acc = []
    ls_eval_acc = []
    ls_train_loss = []
    ls_eval_loss = []
    ls_eval_avg_class_acc = []
    with open(src+"/log_train.txt") as f:
        log = f.readlines()
        log = [line for line in log if not line.startswith('Model saved')]

        # epoch 0
        ls_train_loss.append(line_to_float(log[2]))
        ls_train_acc.append(line_to_float(log[3]))
        ls_eval_loss.append(line_to_float(log[4]))
        ls_eval_acc.append(line_to_float(log[5]))
        ls_eval_avg_class_acc.append((line_to_float(log[6])))

        # epoch 1-N
        for i in range(int((len(log)-5)/6)):
            ls_train_loss.append(line_to_float(log[8 + i * 6]))
            ls_train_acc.append(line_to_float(log[9 + i * 6]))
            ls_eval_loss.append(line_to_float(log[10 + i * 6]))
            ls_eval_acc.append(line_to_float(log[11 + i * 6]))
            ls_eval_avg_class_acc.append((line_to_float(log[12 + i * 6])))
    num_epoch = len(ls_train_acc)

    # Plot results
    fig, axs = plt.subplots(2, 1, layout='constrained', sharex=True)

    axs[0].plot(np.arange(num_epoch), ls_train_acc, label='train')
    axs[0].plot(np.arange(num_epoch), ls_eval_acc, label='eval')
    axs[0].plot(np.arange(num_epoch), ls_eval_avg_class_acc, label='avg_class')
    axs[0].set_title('Accuracy')
    axs[0].set_ylabel('Accuracy value [-]')
    axs[0].legend()

    axs[1].plot(np.arange(num_epoch), ls_train_loss, label='train')
    axs[1].plot(np.arange(num_epoch), ls_eval_loss, label='eval')
    axs[1].set_title('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss value [-]')
    axs[1].legend()

    if do_show:
        plt.show()

    if do_save:
        plt.savefig(src+'/acc_loss_evolution.png')


def show_confusion_matrix(src, class_labels, do_save=True, do_show=False):
    """
        plots the confusion matrix as and image
        :param y_true: list of the GT label of the models
        :param y_pred: List of the predicted label of the models
        :param classes: List of strings containing the label tags
        :param normalize: bool indicating if to normalize the confusion matrix
        :param cmap: colormap to use for plotting
        :return: None (just plots)
        """
    with open(f'{src}/res_eval.pickle', 'rb') as file:
        dict_data = pickle.load(file)
    y_true = dict_data['trues']
    y_pred = dict_data['preds']
    n_classes = len(class_labels)
    conf_mat = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=range(0, n_classes))
    """if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]"""

    fig = plt.figure()
    plt.imshow(conf_mat, cmap=cm.jet)
    ax = plt.gca()
    ax.set_title('Confusion Matrix')

    # Write the labels for each row and column
    if class_labels is not None:
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels, rotation=90, fontsize=8)
        plt.yticks(tick_marks, class_labels, fontsize=8)

    # Write the values in the center of the cell
    # thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        """plt.text(j, i, conf_mat[i, j],
                 horizontalalignment="center", fontsize=10,
                 color="white" if conf_mat[i, j] > thresh else "black")"""
        plt.text(j, i, conf_mat[i, j],
                 horizontalalignment="center", fontsize=8,
                 color="white")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if do_save:
        # plt.savefig(filename +'.pdf',format='pdf', bbox_inches='tight', dpi=1000)
        plt.savefig(src + '/confusion_matrix2.png')
    if do_show:
        plt.show()


if __name__ == '__main__':
    src = "./log/modelnet3/3dmfv_net_cls/grid5_log_trial/1"
    show_log_train(src)
