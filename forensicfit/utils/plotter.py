from email.mime import image
import itertools
from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from .. import HAS_TENSORFLOW
from scipy import stats
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from matplotlib.ticker import MaxNLocator, MultipleLocator

# plt.rcParams["font.family"] = "Arial"
# plt.rc("font", size=11)  # controls default text sizes
# plt.rc("axes", titlesize=14)  # fontsize of the axes title
# plt.rc("axes", labelsize=14)  # fontsize of the x and y labels
# plt.rc("xtick", labelsize=12)  # fontsize of the tick labels
# plt.rc("ytick", labelsize=12)  # fontsize of the tick labels
# plt.rc("legend", fontsize=12)  # legend fontsize


def get_figure_size(dpi: Tuple[int, int],
                    image_shape: Tuple[int, int], 
                    zoom=4,
                    margin=4) -> Tuple[float, float]:
    dpi = np.array(dpi, dtype=np.float_)
    image_shape = np.array(image_shape, dtype=np.float_)
    fig_size = np.flip(image_shape/dpi)*zoom
    # print(fig_size, dpi, image_shape)
    return fig_size+margin
    
def plot_coordinate_based(coordinates: npt.ArrayLike,
                          slopes: npt.ArrayLike,
                          stds: npt.ArrayLike,
                          mode: str = None,
                          ax: Axes = None,
                          plot_slope: bool=True,
                          plot_error_bars: bool=True,
                          plot_edge: bool=True,
                          show: bool=True,
                          **kwargs) -> Axes:
    if ax is None:
        fig = plt.figure()
        ax = fig.subplot(111)
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
    ax.set_facecolor('black')
    color=['cyan', 'magenta', 'black', 'orange', 'red', 'blue', 'green']
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
            ax.plot(x, y, color='green')
        color=['red']
    fig = ax.get_figure()
    large_dim = fig.get_size_inches().max()
    marker_size = large_dim/4
    if plot_error_bars:
        ax.errorbar(coordinates[:, 0],
                    coordinates[:, 1],
                    xerr=stds,
                    ecolor='cyan',
                    color=np.random.choice(color, size=(1,),)[0],
                    ms=marker_size,
                    fmt='o')
    else:
        ax.scatter(coordinates[:, 0],
                    coordinates[:, 1],
                    s=marker_size*4,
                    color=np.random.choice(color, size=(1,),)[0],
                    marker='o')
    if plot_edge:
        ax.plot(coordinates[:, 0],
                coordinates[:, 1],
                color='#f3ff6b')
    # ax.set_ylim(min(coordinates[:, 1]),max(coordinates[:, 1]))            
    # xmin = min(coordinates[:, 0])
    # xmax = max(coordinates[:, 0])
    # ax.set_xlim(xmin-abs(xmin)*0.9, xmax+abs(xmax)*1.1)
    return ax
    

def plot_pair(obj_1: Any,
              obj_2: Any,
              text: List[str] = None,
              which: str='boundary',
              mode: str = None,
              savefig: str = None,
              cmap: str='gray',
              show: bool = True,
              figsize: tuple = None,
              labels: List[str] = None,
              title: str = None,
              zoom:int=4,
              **kwargs,
              ) -> Tuple[Figure, Axes]:
    if text is not None:
        n_columns = 3
    else:
        n_columns = 2
    if 'bin_based' in which and mode == 'individual_bins':
        n_bins = max(
            obj_1.metadata['analysis']['bin_based']['n_bins'],
            obj_1.metadata['analysis']['bin_based']['n_bins']
        )
        x = obj_1.shape[0]
        y = obj_1.metadata['analysis']['bin_based']['window_background']
        y += obj_1.metadata['analysis']['bin_based']['window_tape']
        y *= 2
        x *= 2
        figsize = figsize or get_figure_size(obj_1.metadata['dpi'], 
                                             (x, y), 
                                             zoom) #(n_bins/4, 2*n_bins)
        figure = plt.figure(figsize=figsize)
        axes = figure.subplots(
            n_bins, n_columns,
            gridspec_kw={'hspace': 0.0, 'wspace': 0.00})
        ax = [axes[:, 0], axes[:, 1]]
    else:
        figsize = figsize or get_figure_size(obj_1, zoom)
        figure = plt.figure(figsize=figsize)
        axes = figure.subplots(1, n_columns)
        ax = [axes[0], axes[1]]
    for i, obj in enumerate([obj_1, obj_2]):
        axi = obj.plot(which, ax=ax[i], mode=mode, cmap=cmap, **kwargs)
        for a in axi:
            a.set_xticklabels([])
            a.set_yticklabels([])
        if labels is not None:
            axi.set_title(labels[i])
    if title is not None:
        figure.suptitle(title)
    if text is not None:
        for i, tex in enumerate(text):
            axes[i, 2].text(0.5, 0.5, tex)
            axes[i, 2].xaxis.set_visible(False)
            axes[i, 2].yaxis.set_visible(False)
    if show:
        plt.show()
    if savefig is not None:
        plt.savefig(savefig)
    return figure, ax

def plot_pairs(objs: List[Any],
              text: List[str] = None,
              which: str='boundary',
              mode: str = None,
              savefig: str = None,
              cmap: str='gray',
              show: bool = False,
              figsize: tuple = None,
              labels: List[str] = None,
              title: str = None,
              zoom:int=4,
              **kwargs,
              ) -> Tuple[Figure, Axes]:
    n_objs = len(objs)
    if text is not None:
        n_columns = n_objs + 1
    else:
        n_columns = n_objs
    if 'bin_based' in which and mode == 'individual_bins':
        n_bins = max(
            [x.metadata['analysis']['bin_based']['n_bins'] for x in objs]
            )
        overlap = objs[0].metadata['analysis']['bin_based']['overlap']
        X = objs[0].shape[0] + overlap*(n_bins-2)
        Y = 0
        for obj in objs:
            for window in ['window_background', 'window_tape']:
                Y += obj.metadata.analysis['bin_based'][window]
        if text is not None:
            Y += Y/n_objs
        figsize = figsize or get_figure_size(objs[0].metadata.dpi,
                                             (X, Y), 
                                             zoom
                                             )
        figure, axes = plt.subplots(n_bins, n_columns, 
                                    figsize=figsize
                                    )
        # figure = plt.figure(figsize=figsize)
        #  = figure.subplots(
            # 
        #     gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
        ax = [axes[:, i] for i in range(n_objs)]
    else:
        X = objs[0].shape[0]
        Y = objs[0].shape[1]*n_objs
        figsize = figsize or get_figure_size(objs[0].metadata.dpi,
                                             (X, Y), 
                                             zoom)
        figure = plt.figure(figsize=figsize)
        axes = figure.subplots(1, n_columns, 
                               gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
        ax = [axes[i] for i in range(n_objs)]
    print(f'Created a figure with the dimensions: {figsize[0]}, {figsize[1]}')
    for i, obj in enumerate(objs):
        axi = obj.plot(which, ax=ax[i], mode=mode, cmap=cmap, **kwargs)
        # if not isinstance(axi, list): 
        #     axi = [axi]
        for a in axi:
            a.set_xticklabels([])
            a.set_yticklabels([])
        if labels is not None:
            axi.set_title(labels[i])
    if title is not None:
        figure.suptitle(title)
    if text is not None:
        for i, tex in enumerate(text):
            axes[i, -1].text(0.1, 0.1, tex)
            axes[i, -1].xaxis.set_visible(False)
            axes[i, -1].yaxis.set_visible(False)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    if show:
        plt.show()

    return figure, ax

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