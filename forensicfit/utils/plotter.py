import itertools
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.axes import Axes


plt.rcParams["font.family"] = "Arial"
plt.rc("font", size=11)  # controls default text sizes
plt.rc("axes", titlesize=14)  # fontsize of the axes title
plt.rc("axes", labelsize=14)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=12)  # fontsize of the tick labels
plt.rc("ytick", labelsize=12)  # fontsize of the tick labels
plt.rc("legend", fontsize=12)  # legend fontsize


def plot_pair(obj_1: Any,
              obj_2: Any,
              text: List[str] = None,
              which: str = None,
              mode: str = None,
              savefig: str = None,
              cmap: str = 'gray',
              show: bool = True,
              figsize: tuple = None,
              labels: List[str] = None,
              **kwargs,
              ):
    if text is not None:
        n_columns = 3
    else:
        n_columns = 2
    if which == 'bin_based' and mode == 'individual_bins':
        n_bins = max(
            obj_1.metadata['analysis']['bin_based']['n_bins'],
            obj_1.metadata['analysis']['bin_based']['n_bins']
        )
        figsize = figsize or (n_bins/4, 2*n_bins)
        figure = plt.figure(figsize=figsize)
        axes = figure.subplots(
            n_bins, n_columns,
            gridspec_kw={'hspace': 0.1, 'wspace': 0.001})
        ax = [axes[:, 0], axes[:, 1]]
    else:
        figsize = figsize or (20, 10)
        figure = plt.figure(figsize=figsize)
        axes = figure.subplots(1, n_columns)
        ax = [axes[0], axes[1]]
    for i, obj in enumerate([obj_1, obj_2]):
        axi = obj.plot(which, ax=ax[i], mode=mode, cmap=cmap, **kwargs)
        if labels is not None:
            axi.set_title(labels[i])
    if text is not None:
        for i, tex in enumerate(text):
            axes[i, 2].text(0.5, 0.5, tex)
            axes[i, 2].xaxis.set_visible(False)
            axes[i, 2].yaxis.set_visible(False)
    if show:
        plt.show()
    if savefig is not None:
        plt.savefig(savefig)
    return


def plot_confusion_matrix(matrix: np.asarray,
                          class_names: np.asarray,
                          title: str = 'Confusion matrix',
                          cmap: str = 'Blues',
                          normalize: bool = False,
                          savefig: str = None,
                          ax: Axes = None,
                          show: bool = True,
                          colorbar: bool = True):

    accuracy = np.trace(matrix) / float(np.sum(matrix))
    misclass = 1 - accuracy

    cmap = plt.get_cmap(cmap)

    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111)
    im = ax.imshow(matrix, cmap=cmap)
    ax.set_title(title)
    if colorbar:
        plt.colorbar(im)

    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks, class_names)  # , rotation=45)
        ax.set_yticks(tick_marks, class_names)

    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    thresh = matrix.max() / 1.5 if normalize else matrix.max() / 2
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        if normalize:
            ax.text(j, i, "{:0.4f}".format(matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")
        else:
            ax.text(j, i, "{:,}".format(matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")

    # ax.set_ylabel('True label')
    # ax.set_xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(
        # accuracy, misclass))
    if savefig:
        plt.savefig(savefig)
    if show:
        plt.show()
    return ax

def plot_kde_distribution(distribution: np.ndarray,
                            color: str = 'blue',
                            opacity: float = 1.0,
                            label: str = '',
                            ax: Axes = None,
                            savefig: str = None,
                            fill_curve: bool = True,
                            show: bool = True):
    span = np.linspace(distribution.min(), distribution.max())
    kernel = stats.gaussian_kde(distribution)
    estimated_distribution = kernel(span)
    if ax is None:
        plt.figure(16, 9)
        ax = plt.subplot(111)
    ax.plot(span, estimated_distribution, color=color, label=label)
    if fill_curve:
        ax.fill_between(span, estimated_distribution, color=color, alpha=opacity)
    ax.set_xlim(0, 1)
    if savefig:
        plt.savefig(savefig)
    if show:
        plt.show()
    return ax
        
def plot_hist_distribution(distribution: np.ndarray,
                           color: str = 'blue',
                           opacity: float = 1.0,
                           label: str = '',
                           ax: Axes = None,
                           savefig: str = None,
                           show: bool = True):
    
    if ax is None:
        plt.figure(16, 9)
        ax = plt.subplot(111)
    ax.hist(distribution, color=color, label=label, alpha=opacity)
    ax.set_xlim(0, 1)
    if savefig:
        plt.savefig(savefig)
    if show:
        plt.show()
    return ax