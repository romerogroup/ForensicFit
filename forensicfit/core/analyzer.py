# -*- coding: utf-8 -*-

from typing import List
import cv2
from matplotlib import gridspec, pylab as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
import numpy as np
from scipy.stats import norm
from abc import ABCMeta, abstractmethod
import io
from collections.abc import Mapping
from .metadata import Metadata
from ..utils.image_tools import IMAGE_EXTENSIONS


class Analyzer:
    
    __metaclass__ = ABCMeta
    """Class containing all future analyzers
    """
    
    def __init__(self, **kwargs):
        """
        

        Returns
        -------
        None.

        """
        
        self.image = None
        self.values = {}
        self.metadata = Metadata({'mode': 'analysis', 
                                  'label': None, 
                                  'material': None})
        self.metadata.update(kwargs)

    def plot_boundary(self, savefig = None, color='r', ax = None, show=False):
        """
        This function plots the detected boundary of the image. 

        Parameters
        ----------
        savefig : str, optional
            path to save the plot. The default is None.
        
        color : str, optional
            Color of the boundary. The default is 'r'

        ax : str, optional
            Re. The default is None

        show: bool
            Controls whether to show the image. Default is False

        

        Returns
        -------
        None.

        """
        if ax is None:
            plt.figure(figsize = (16, 9))
            ax = plt.subplot(111)
        ax.plot(self.boundary[:, 0], self.boundary[:, 1], c = color)
        if savefig is not None:
            plt.savefig(savefig)
        elif show:
            plt.show()
        return ax

        
    def plot(self,
             which: str, 
             cmap: str='gray', 
             savefig: str = None, 
             ax: Axes or List[Axes]= None, 
             show: bool = False,
             mode: str = None, 
             **kwargs):
       
        """
        Parameters
        ----------
        which : TYPE
            DESCRIPTION.
        cmap : TYPE, optional
            DESCRIPTION. The default is 'gray'.
        savefig : TYPE, optional
            DESCRIPTION. The default is None.
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        reverse_x : TYPE, optional
            DESCRIPTION. The default is False.
        savefig : str
            Path and name to save the image. Default is None

        ax : str
            Path and name to save the image. Default is None

        revere_x: bool

        show: bool
            Controls whether to show the image. Default is False

        plot_gaussian: bool
            plots the gaussian smearing method. Default is False

        plot_errorbar: bool
            plots the gaussian smearing method. Default is False
        Returns
        -------
        None.

        """
        
        if which == "coordinate_based":
            if ax is None:
                plt.figure(figsize=(3, 10))
                ax = plt.subplot(111)
            if mode == "gaussian":
                dy = (self.ymax-self.ymin)/len(self[which])
                # norm = Normalize(vmin, vmax)
                cmap=plt.get_cmap('gray')
                for ig in self[which]:
                    x = np.linspace(ig[0]-3*ig[2], ig[0]+3*ig[2])
                    dx = x[2]-x[1]
                    y = np.ones_like(x)*ig[1]
                    y_prime = norm.pdf(x, ig[0], ig[2])
                    y_prime /= sum(y_prime)/dx
                    colors = cmap(y_prime)
                    y_prime*=dy
                    ax.fill_between(x, y, y+y_prime, cmap='gray')
            
            elif mode == "errorbar":
                ax.errorbar(self[which][:, 0],
                            np.flip(self[which][:, 1]),
                            xerr= self[which][:, 2],
                            ecolor='blue',
                            color='red',
                            markersize=0.5,
                            fmt='o')

            else:
                ax.scatter(self[which][:, 0],
                        np.flip(self[which][:, 1]),
                        c='red',
                        s=1)
            ax.set_ylim(min(self[which][:, 1]),max(self[which][:, 1]))            
            ymin = min(self[which][:, 0])
            ax.set_xlim(ymin-abs(ymin)*0.9, max(self[which][:, 0])*1.1)
        elif which == 'boundary':
            ax = self.plot('image', cmap=cmap, ax=ax)
            ax = self.plot_boundary(ax=ax)
        elif which == 'bin_based':
            if mode == 'individual_bins':
                dynamic_positions = self.metadata[
                    'analysis']['bin_based']['dynamic_positions']
                n_bins = self.metadata['analysis']['bin_based']['n_bins']
                if ax is None:
                    figure = plt.figure(figsize=(10, 20))
                    ax = figure.subplots(
                        n_bins, 1,
                        sharex=True, 
                        gridspec_kw={'hspace':2e-2})
                if isinstance(ax, list):
                    assert len(ax) >= n_bins, 'Number of Axes provided ' \
                        "smaller than the number of bins"
                else:
                    ax = [ax]
                bins = self['bin_based']
                for i, seg in enumerate(bins):
                    ax[i].imshow(seg, cmap=cmap)
                    ax[i].xaxis.set_visible(False)
                    ax[i].yaxis.set_visible(False)
                ax = ax[-1]
            else:
                if ax is None:
                    plt.figure(figsize=(16, 9))
                    ax = plt.subplot(111)
                dynamic_positions = self.metadata[
                    'analysis'][which]['dynamic_positions']
                colors = [
                    'red', 'blue', 'green', 'cyan', 'magenta'
                    ]*len(dynamic_positions)
                styles = [
                    'solid', 'dashed', 'dotted', 'dashdot'
                    ]*len(dynamic_positions)
                xs = []
                for i, seg in enumerate(dynamic_positions):
                    y1 = seg[1][0]
                    y2 = seg[1][1]
                    x1 = seg[0][0]
                    x2 = seg[0][1]
                    xs.append(x1)
                    xs.append(x2)
                    ax.plot([x1, x1], [y1, y2], color=colors[i], 
                            linestyle=styles[i], linewidth=1)
                    ax.plot([x2, x2], [y1, y2], color=colors[i], 
                            linestyle=styles[i], linewidth=1)
                    ax.plot([x1, x2], [y1, y1], color=colors[i], 
                            linestyle=styles[i], linewidth=1)
                    ax.plot([x1, x2], [y2, y2], color=colors[i], 
                            linestyle=styles[i], linewidth=1)
                ax = self.plot('image', ax=ax, cmap=cmap)
        else:
            if ax is None:
                plt.figure(figsize = (16, 9))
                ax = plt.subplot(111)
            ax.imshow(self[which], cmap=cmap)
            ax.set_xlim(0, self[which].shape[1])
        if which != 'coordinate_based':
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        if savefig is not None:
            plt.savefig(savefig)
            return ax
        if show:
            plt.show()
        else:
            return ax

    def show(self, which, wait=0):
        """
        

        Parameters
        ----------
        which : TYPE
            DESCRIPTION.
        wait : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        cv2.imshow(which, self[which])
        cv2.waitKey(wait)
        cv2.destroyAllWindows()

    @abstractmethod
    def load_dict(self):
        pass

    @abstractmethod
    def from_dict(self):
        pass

    @classmethod
    def from_buffer(cls, 
                    buffer: bytes, 
                    metadata: dict,
                    ext: str='.png',
                    allow_pickle: bool=False):
        """receives an io byte buffer with the corresponding metadata and 
        creates the image class

        Parameters
        ----------
        buffer : io.BytesIO
            _description_
        metadata : dict
            _description_
        allow_pickle : bool, optional
            _description_, by default False
        """        
        if 'ext' in metadata:
            ext = metadata['ext']
        if ext in IMAGE_EXTENSIONS:
            image = cv2.imdecode(np.frombuffer(buffer, np.uint8), -1)
            return cls.from_dict(image, metadata)
    
    def to_buffer(self, ext: str = '.png'):
        if ext in IMAGE_EXTENSIONS:
            is_success, buffer = cv2.imencode(ext, self.image)
            output = io.BytesIO(buffer)
        else:
            raise ValueError("Extension not supported")
        return output.getvalue()
        
    def __contains__(self, x):
        return x in self.values

    def __getitem__(self, x):
        return self.values.__getitem__(x)

    def __iter__(self):
        return self.values.__iter__()

    def __len__(self):
        return self.values.__len__()
