import itertools
from typing import Any, List, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from .. import HAS_TENSORFLOW
from scipy import stats
from matplotlib.axes import Axes
from pathlib import Path
from matplotlib.ticker import MaxNLocator, MultipleLocator

plt.rcParams["font.family"] = "Arial"
plt.rc("font", size=11)  # controls default text sizes
plt.rc("axes", titlesize=14)  # fontsize of the axes title
plt.rc("axes", labelsize=14)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=12)  # fontsize of the tick labels
plt.rc("ytick", labelsize=12)  # fontsize of the tick labels
plt.rc("legend", fontsize=12)  # legend fontsize


def get_figure_size(obj, zoom=4):
    dpi = (1000, 1000)
    if 'dpi' in obj.metadata:
        dpi = np.array(obj.metadata.dpi, dtype=np.float_)
    figsize = np.flip(obj.shape[:2]/dpi)*zoom
    return figsize

def plot_coordinate_based(coordinates: npt.ArrayLike,
                          slopes: npt.ArrayLike,
                          stds: npt.ArrayLike,
                          mode: str = None,
                          ax: Axes = None,
                          plot_slope: bool=False,
                          show: bool=True):
    n_points = len(coordinates)
    if ax is None:
        
        plt.figure(figsize=(16, 9))
        ax = plt.subplot(111)
    # if mode == "gaussians":
    #     dy = (self.ymax-self.ymin)/n_points
    #     # norm = Normalize(vmin, vmax)
    #     cmap=plt.get_cmap('gray')   
    #     coordinates[:, 1] = np.flip(coordinates[:, 1])
    #     for i, ig in enumerate(coordinates):
    #         x = np.linspace(ig[0]-3*stds[i], ig[0]+3*stds[i])
    #         dx = (x[2]-x[1])
    #         y = np.ones_like(x)*ig[1]
    #         y_prime = norm.pdf(x, ig[0], stds[i])
    #         y_prime /= sum(y_prime)/dx
    #         colors = cmap(y_prime)
    #         y_prime*=dy
    #         ax.fill_between(x, y, y+y_prime, cmap='gray')
    #         ax.scatter(coordinates[:, 0],
    #             coordinates[:, 1],
    #             c='black',
    #             s=0.01)
    color=['red', 'blue', 'green', 'cyan', 'magenta', 'black', 'orange']
    if plot_slope:
        dy = coordinates[1, 1] - coordinates[0, 1]
        dy *= 0.7
        for i, iseg in enumerate(slopes):
            m = iseg[0]
            b0 = iseg[1]
            y0 = coordinates[i, 1]
            y_min = y0 - dy/2
            y_max = y0 + dy/2
            y = np.linspace(y_min, y_max, 100)
            x = y/m - b0/m
            ax.plot(x, y, color='blue')
        color=['red']
    if mode == "error_bars":
        ax.errorbar(coordinates[:, 0],
                    np.flip(coordinates[:, 1]),
                    xerr=stds,
                    ecolor='blue',
                    color='red',
                    markersize=0.5,
                    fmt='o')
    else:
        ax.scatter(coordinates[:, 0],
                   coordinates[:, 1],
                s=1,
                color=np.random.choice(color, size=(1,),)[0])
    # ax.set_ylim(min(coordinates[:, 1]),max(coordinates[:, 1]))            
    xmin = min(coordinates[:, 0])
    xmax = max(coordinates[:, 0])
    # ax.set_xlim(xmin-abs(xmin)*0.9, xmax+abs(xmax)*1.1)
    return ax
    

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

if HAS_TENSORFLOW:
    import tensorflow as tf
    def plot_metrics(history: tf.keras.callbacks.History,
                    metrics: List[Union[tf.keras.metrics.Metric, str]],
                    has_valid: bool=True,
                    savefig: str = None):
        """Plots the given 'metric' from 'history'.
        """
        n_plots = len(metrics)
        n_rows = n_plots//2 + n_plots%2
        fig = plt.figure(figsize=((14, n_rows*2+10)))
        axes = fig.subplots(n_rows, 2, sharex=True)
        for i, metric in enumerate(metrics):
            if 'keras' in str(type(metric)):
                name = metric.name
                ylabel = metric.__str__().split('(')[0]
            elif isinstance(metric, str):
                name = metric
                ylabel = metric.capitalize()
            EPOCHS = len(history[name])
            x = np.arange(1, EPOCHS+1)
            ax = axes[i//2, i%2]
            ax.plot(x, history[name], color='blue')
            if has_valid:
                ax.plot(x, history["val_" + name], color='red')
                ax.legend(["train", "validation"])#, loc="upper left")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.tick_params(axis="x", direction="inout")
            ax.tick_params(which="minor", axis="x", direction="in") 
            ax.tick_params(axis="y", direction="inout")
            ax.set_ylabel(ylabel)
            if i//2 == n_rows - 1:
                ax.set_xlabel("epoch")
            ax.set_xlim(1, EPOCHS)
        plt.tight_layout()
        if savefig is not None:
            savefig = Path(savefig)
            plt.savefig(savefig.absolute().as_posix())
        plt.show()