import itertools
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(handler)


def init_confusion_matrix(conf_mx, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function pretty prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(conf_mx, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation='vertical')
    plt.yticks(tick_marks, classes, rotation=0)

    if normalize:
        conf_mx = conf_mx.astype('float') / conf_mx.sum(axis=1)[:, np.newaxis]
        log.info('Plotting normalized confusion matrix')
    else:
        log.info('Plotting confusion matrix without normalization')

    thresh = conf_mx.max() / 2.
    for i, j in itertools.product(range(conf_mx.shape[0]), range(conf_mx.shape[1])):
        plt.text(j, i, conf_mx[i, j],
                 horizontalalignment="center",
                 color="white" if conf_mx[i, j] > thresh else "black")

    #  plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confusion_matrix(conf_mx, class_names):
    # TODO: load targets labels from config
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    init_confusion_matrix(conf_mx, classes=class_names, title='Confusion matrix')
    plt.show()
