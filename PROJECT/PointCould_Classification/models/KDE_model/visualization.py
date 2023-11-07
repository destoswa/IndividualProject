from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.cm as cm
import itertools
import pandas as pd


def show_log_train(data_src, target_src, do_save=True, do_show=False):
    data = pd.read_csv(data_src, delimiter=';')
    ls_train_acc = data['train_acc'].to_list()
    ls_train_loss = data['train_loss'].to_list()
    ls_test_acc = data['test_acc'].to_list()
    ls_test_class_acc = data['test_class_acc'].to_list()
    ls_test_loss = data['test_loss'].to_list()
    num_epoch = len(ls_train_acc)

    # Plot results
    fig, axs = plt.subplots(2, 1, layout='constrained', sharex=True)

    # plot accuracies
    axs[0].plot(np.arange(num_epoch), ls_train_acc, label='train')
    axs[0].plot(np.arange(num_epoch), ls_test_acc, label='eval')
    axs[0].plot(np.arange(num_epoch), ls_test_class_acc, label='eval class')
    axs[0].set_title('Accuracy')
    axs[0].set_ylabel('Accuracy value [-]')
    axs[0].set_ylim(None, 1.0)
    axs[0].legend(loc='upper left')

    # plot losses
    axs[1].plot(np.arange(num_epoch), ls_train_loss, label='train')
    axs[1].plot(np.arange(num_epoch), ls_test_loss, label='eval')
    axs[1].set_title('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss value [-]')
    axs[1].legend()

    if do_save:
        plt.savefig(target_src + '/acc_loss_evolution.png')

    if do_show:
        plt.show()


def show_confusion_matrix(target_src, y_pred, y_true, class_labels, epoch=0, do_save=True, do_show=False):
    """
        plots the confusion matrix as and image
        :param target_src : location of saved image
        :param y_true: list of the GT label of the models
        :param y_pred: List of the predicted label of the models
        :param class_labels: List of strings containing the label tags
        :param epoch: number of the epoch of training which provided the results
        :param do_save: saves the image
        :param do_show: shows the image
        :return: None (just plots)
        """
    """df_data = pd.read_csv(src, sep=';')
    y_true = df_data['target'].to_list()
    y_pred = df_data['pred'].to_list()"""
    n_classes = len(class_labels)
    conf_mat = confusion_matrix(y_true, y_pred, labels=range(0, n_classes))

    fig = plt.figure()
    plt.imshow(conf_mat, cmap=cm.jet)
    ax = plt.gca()
    ax.set_title('Confusion Matrix - epoch ' + str(epoch + 1))

    # Write the labels for each row and column
    if class_labels is not None:
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels, rotation=90, fontsize=8)
        plt.yticks(tick_marks, class_labels, fontsize=8)

    # Write the values in the center of the cell
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, conf_mat[i, j],
                 horizontalalignment="center", fontsize=8,
                 color="white")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.tight_layout()

    if do_save:
        plt.savefig(target_src + '/confustion_matrix.png')

    if do_show:
        plt.show()


if __name__ == '__main__':
    src = "./log/cls/Menghao/temp_results"
    version = 0
    best_epoch = 2
    SAMPLE_LABELS = ['Garbage', 'Multi', 'Single']
    """show_log_train(src + '/logs_v' + str(version)+'.csv', src, do_show=True)
    show_confusion_matrix(src + '/confmat_v' + str(version) + '.csv', src, SAMPLE_LABELS, best_epoch, do_show=True)"""
    show_log_train(src + '/logs_v1_nodataaug_100epoch.csv', src, do_show=True)
    show_confusion_matrix(src + '/confmat_v1_nodataaug_100epoch.csv', src, SAMPLE_LABELS, best_epoch, do_show=True)
