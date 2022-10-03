# -*- coding: utf-8 -*-

import io
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Union

import cv2
from forensicfit.utils import plotter
import numpy as np
from matplotlib import pylab as plt
from matplotlib.axes import Axes
from scipy.stats import norm

from forensicfit import utils

from ..utils import copy_doc, image_tools, plotter
from .metadata import Metadata

IMAGE_EXTENSIONS = image_tools.IMAGE_EXTENSIONS

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

    @copy_doc(image_tools.exposure_control)
    def exposure_control(self, mode:str='equalize_hist', **kwargs):
        self.original_image = self.image
        self.image = image_tools.exposure_control(self.image)
        self.metadata['exposure_control'] = mode
        if len(kwargs) != 0:
            for key in kwargs:
                self.metadata[key] = kwargs[key]
        return

    @copy_doc(image_tools.apply_filter)
    def apply_filter(self, mode:str, **kwargs):
        self.original_image = self.image
        self.image = image_tools.apply_filter(self['gray_scale'], **kwargs)
        self.metadata['filter'] = mode
        if len(kwargs) != 0:
            for key in kwargs:
                self.metadata[key] = kwargs[key]
        return

    def plot_boundary(self, 
                      savefig: Union[str, Path] = None, 
                      color: str='red', 
                      ax: Axes = None, 
                      show: bool=False):
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
             zoom: int=4,
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
                figsize = plotter.get_figure_size(self.metadata['dpi'],
                                                  self.shape[:2], 
                                                  zoom)
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)
            coordinates = self['coordinate_based']['coordinates']
            stds = self['coordinate_based']['stds']
            slopes = self['coordinate_based']['slopes']
            ax = utils.plotter.plot_coordinate_based(coordinates,
                                                slopes,
                                                stds,
                                                mode,
                                                ax, **kwargs)
            # ax.set_xlim(0, self.image.shape[1])
            # ax.set_ylim(0, self.image.shape[0])
            ax.invert_yaxis()
        elif which == 'boundary':
            ax = self.plot('image', cmap=cmap, ax=ax)
            ax = self.plot_boundary(ax=ax)
        elif which in ['bin_based+coordinate_based',
                       'coordinate_based+bin_based']:
            if mode == 'individual_bins':
                dynamic_positions = np.array(self.metadata[
                    'analysis']['bin_based']['dynamic_positions'])
                xmin = min(dynamic_positions[:, 0, 0])
                xmax = max(dynamic_positions[:, 0, 1])
                
                
                
                n_bins = self.metadata['analysis']['bin_based']['n_bins']
                if ax is None:
                    figure = plt.figure(figsize=(5, 2*n_bins))
                    ax = figure.subplots(
                        n_bins, 1,
                        sharex=True, 
                        gridspec_kw={'hspace':2e-2})
                
                elif isinstance(ax, list):
                    assert len(ax) >= n_bins, 'Number of Axes provided ' \
                        "smaller than the number of bins"
                if n_bins == 1: ax=[ax]

                for i, i_bin in enumerate(self[which]):
                    coordinates = i_bin['coordinates']
                    stds = i_bin['stds']
                    slopes = i_bin['slopes']
                    ax[i] = utils.plotter.plot_coordinate_based(coordinates,
                                                    slopes,
                                                    stds,
                                                    mode,
                                                    ax[i], **kwargs)
                    dy = coordinates[1, 1] - coordinates[0, 1]
                    y_min, y_max = min(coordinates[:, 1]), max(coordinates[:, 1])
                    ax[i].set_ylim(y_min-dy, y_max+dy)
                    ax[i].invert_yaxis()
                ax = ax[-1]
                
                
                ax.set_xlim(xmin, xmax)
                # ax.set_ylim(xmin, xmax)

            else:
                if ax is None:
                    plt.figure(figsize=(16, 9))
                    ax = plt.subplot(111)
                for i_bin in self[which]:
                    coordinates = i_bin['coordinates']
                    stds = i_bin['stds']
                    slopes = i_bin['slopes']
                    ax = utils.plotter.plot_coordinate_based(coordinates,
                                                    slopes,
                                                    stds,
                                                    mode,
                                                    ax, **kwargs)

                ax.set_ylim(0, self.image.shape[0])
                ax.set_xlim(0, self.image.shape[1])
                ax.invert_yaxis()
                
            
        elif which in [
            'bin_based', 
            'bin_based+max_contrast', 
            'max_contrast+bin_based']:
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
                
                elif isinstance(ax, list):
                    assert len(ax) >= n_bins, 'Number of Axes provided ' \
                        "smaller than the number of bins"
                if n_bins == 1: ax=[ax]
                
                bins = self[which]
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
                    'analysis']['bin_based']['dynamic_positions']
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
                if 'max_contrast' in which:
                    ax = self.plot('edge_bw', ax=ax, cmap=cmap)
                else:
                    ax = self.plot('image', ax=ax, cmap=cmap)
        else:
            if ax is None:
                figsize = plotter.get_figure_size(self.metadata['dpi'],
                                                  self.shape[:2], 
                                                  zoom)
                plt.figure(figsize = figsize)
                ax = plt.subplot(111)
            ax.imshow(self[which], cmap=cmap)
            ax.set_xlim(0, self[which].shape[1])
        if 'coordinate_based' not in which:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        if savefig is not None:
            plt.savefig(savefig)
            return ax
        if show:
            plt.show()
        else:
            return ax

    @abstractmethod
    def load_dict(self):
        pass

    @property
    def shape(self):
        return self.image.shape

    @abstractmethod
    def from_dict(self):
        pass

    @classmethod
    def from_buffer(cls, 
                    buffer: bytes, 
                    metadata: dict,
                    ext: str='.png'):
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

    def __repr__(self) -> str:
        self.plot(which='boundary', show=True)
        ret = self.metadata.__str__()
        return ret