import numpy as np
import os
import sys
import matplotlib.pyplot as plt

plt.switch_backend('agg')

sourceDomainPath = '../SourceDomainData/'
targetDomainPath = '../TargetDomainData/'
experimentalPath = '../experiment_data/'
pulmonary_category = {0: 'airplane',
                      1: 'automobile',
                      2: 'bird',
                      3: 'cat',
                      4: 'deer',
                      5: 'dog',
                      6: 'frog',
                      7: 'horse',
                      8: 'ship',
                      9: 'truck'}


def save2file(message, checkpointPath, model_name):
    if not os.path.isdir(checkpointPath):
        os.makedirs(checkpointPath)
    logfile = open(checkpointPath + model_name + '.txt', 'a+')
    print(message)
    print(message, file=logfile)
    logfile.close()


def plot_acc_loss(checkpointPath, epoch, epoch_plt, train_acc_plt, train_loss_plt, val_acc_plt, val_loss_plt):
    plt.figure(figsize=(10.24, 7.68))
    plt.plot(epoch_plt, train_acc_plt, linewidth=1.0, linestyle='-', label='training_accuracy')
    plt.plot(epoch_plt, val_acc_plt, linewidth=1.0, color='red', linestyle='--',
             label='validation_accuracy')
    plt.title('Accuracy')
    plt.ylim([0.6, 1.2])
    plt.xticks([x for x in range(0, epoch + 1, 10)])
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(checkpointPath + 'Accuracy.png')
    plt.close()

    plt.figure(figsize=(10.24, 7.68))
    plt.plot(epoch_plt, train_loss_plt, linewidth=1.0, linestyle='-', label='training_loss')
    plt.plot(epoch_plt, val_loss_plt, color='red', linewidth=1.0, linestyle='--', label='validation_loss')
    plt.title('Loss')
    plt.ylim([0.0, 1.5])
    plt.xticks([x for x in range(0, epoch + 1, 10)])
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig(checkpointPath + 'Loss.png')
    plt.close()
