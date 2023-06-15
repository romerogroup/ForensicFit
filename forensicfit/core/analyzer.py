# -*- coding: utf-8 -*-
"""
analyzer.py

This module contains the Analyzer class, which is responsible for 
handling and analyzing images in the context of the ForensicFit 
application. It uses computer vision techniques for image analysis 
and provides utilities for plotting the results and converting 
images to and from byte buffers.

The module includes the following classes:
- Analyzer: An abstract base class that defines the necessary 
interface for image analysis in the ForensicFit application.

Author: Pedram Tavadze
Email: petavazohi@gmail.com
"""

import io
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Union

import cv2
import matplotlib as mpl
import numpy as np
from matplotlib import pylab as plt
from matplotlib.axes import Axes
from scipy.stats import norm

from ..utils import copy_doc, image_tools, plotter
from .metadata import Metadata

IMAGE_EXTENSIONS = image_tools.IMAGE_EXTENSIONS

class Analyzer:
    """
    Abstract base class that represents an analyzer in the system.

    The Analyzer class is designed to be subclassed by concrete analyzer classes. 
    Each subclass should implement the methods that make sense for that specific 
    type of analyzer.

    This class uses the Abstract Base Classes (ABC) module which enables the 
    creation of a blueprint for other classes. This means you can't create an 
    instance of this class, it is intended to be subclassed. All methods marked 
    with @abstractmethod must be implemented in any concrete (i.e., 
    non-abstract) subclass.

    Attributes
    ----------
    image : np.ndarray
        The image to be analyzed. This attribute is expected to be a numpy 
        array representing the image, but it's initially set to None.
    values : dict
        A dictionary that contains the results of the analysis. The keys are 
        strings describing what each value represents.
    metadata : Metadata
        An instance of the Metadata class, containing metadata related to the 
        analysis. 

    Notes
    -----
    This class is part of a module called "analyzer.py". It serves as the 
    parent class for all future analyzers in the system.
    """
    __metaclass__ = ABCMeta

    
    def __init__(self, **kwargs):
        self.image = None
        self.values = {}
        self.metadata = Metadata({'mode': 'analysis', 
                                  'label': None, 
                                  'material': None})
        self.metadata.update(kwargs)

    # @copy_doc(image_tools.exposure_control)
    def exposure_control(self, mode:str='equalize_hist', **kwargs):
        self.original_image = self.image
        self.image = image_tools.exposure_control(self.image, mode, **kwargs)
        if self.metadata['remove_background']:
            self.image = image_tools.remove_background(self.image, 
                                                       self.largest_contour)
        self.metadata['exposure_control'] = mode
        if len(kwargs) != 0:
            for key in kwargs:
                self.metadata[key] = kwargs[key]
        return

    # @copy_doc(image_tools.apply_filter)
    def apply_filter(self, mode:str, **kwargs):
        self.original_image = self.image
        self.image = image_tools.apply_filter(self['gray_scale'], mode, **kwargs)
        self.metadata['filter'] = mode
        if len(kwargs) != 0:
            for key in kwargs:
                self.metadata[key] = kwargs[key]
        return

    def resize(self, size: tuple = None, dpi: tuple = None):
        """
        Resize the image associated with this analyzer.

        The method allows to resize the image either by providing the desired 
        output size, or by providing the dots per inch (dpi). If both parameters 
        are None, the image will not be modified.

        Parameters
        ----------
        size : tuple, optional
            Desired output size in pixels as (height, width). If provided, this 
            value will be used to resize the image. Default is None.
        dpi : tuple, optional
            Desired dots per inch (dpi) as (horizontal, vertical). If provided and 
            'dpi' key exists in the metadata, this value will be used to compute 
            the new size and then resize the image. Default is None.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If both 'size' and 'dpi' are None.
        """
        if dpi is None and size is not None:
            self.image = image_tools.resize(self.image, size)
            self.values['image'] = self.image
            self.metadata['resize'] = size
            self.metadata['resolution'] = self.image.shape
        elif dpi is not None and 'dpi' in self.metadata:
            dpi = np.array(dpi, dtype=np.int_)
            dpi_old = np.array(self.metadata.dpi, dtype=np.int_)
            ratio = dpi/dpi_old
            size = np.flip((np.array(self.shape)[:2]*ratio).round().astype(int))
            self.image = image_tools.resize(self.image, size)
            self.values['image'] = self.image
            self.metadata['resize'] = size
            self.metadata['resolution'] = self.image.shape
            self.metadata['resolution'] = self.image.shape
            self.metadata['dpi'] = dpi
        else:
            raise ValueError('Please provide either size or dpi')

    def plot_boundary(self, 
                      savefig: Union[str, Path] = None, 
                      color: str='red', 
                      ax: Axes = None, 
                      show: bool=False):
        """
        Plots the detected boundary of the image. 

        Parameters
        ----------
        savefig : Union[str, Path], optional
            Path to save the plot. If provided, the plot will be saved at the 
            specified location. If None, the plot will not be saved. 
            Default is None.
        
        color : str, optional
            Color of the boundary line in the plot. Default is 'red'.

        ax : matplotlib.axes.Axes, optional
            An instance of Axes in which to draw the plot. If None, a new 
            Axes instance will be created. Default is None.

        show: bool, optional
            Controls whether to show the image using matplotlib.pyplot.show() 
            after it is drawn. Default is False.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Axes instance in which the plot was drawn.

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
             savefig: Union[str, Path] = None, 
             ax: Union[Axes, List[Axes]] = None, 
             show: bool = False,
             mode: str = None, 
             **kwargs):
        """
        Plots different kinds of data based on the given parameters.

        Parameters
        ----------
        which : str
            Determines the kind of plot to be created. Possible values include 
            "coordinate_based", "boundary", "bin_based+coordinate_based", 
            "coordinate_based+bin_based", "bin_based", 
            "bin_based+max_contrast", "max_contrast+bin_based" and others.

        cmap : str, optional
            The Colormap instance or registered colormap name. Default is 'gray'.

        zoom : int, optional
            The zoom factor for the plot. Default is 4.

        savefig : str, optional
            Path and name to save the image. If None, the plot will not be saved. 
            Default is None.

        ax : matplotlib.axes.Axes or List[matplotlib.axes.Axes], optional
            An instance of Axes or list of Axes in which to draw the plot. If None, 
            a new Axes instance will be created. Default is None.

        show : bool, optional
            If True, displays the image. Default is False.

        mode : str, optional
            Determines the mode of operation, which affects how the plot is 
            generated. The effect depends on the value of `which`.

        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        ax : matplotlib.axes.Axes or List[matplotlib.axes.Axes]
            The Axes instance(s) in which the plot was drawn.

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
            ax = plotter.plot_coordinate_based(coordinates,
                                                slopes,
                                                stds,
                                                mode,
                                                ax, **kwargs)
            ax.set_xlim(0, self['image'].shape[1])
            # ax.set_xlim(0, self.image.shape[1])
            # ax.set_ylim(0, self.image.shape[0])
            ax.invert_yaxis()
        elif which == 'boundary':
            ax = self.plot('image', cmap=cmap, ax=ax)
            ax = self.plot_boundary(ax=ax)
            ax_ = ax
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
                    ax[i] = plotter.plot_coordinate_based(coordinates,
                                                    slopes,
                                                    stds,
                                                    mode,
                                                    ax[i], **kwargs)
                    dy = coordinates[1, 1] - coordinates[0, 1]
                    y_min, y_max = min(coordinates[:, 1]), max(coordinates[:, 1])
                    # ax[i].xaxis.set_visible(False)
                    # ax[i].yaxis.set_visible(False)
                    # ax[i].set_ylim(y_min-dy, y_max+dy)
                    y1 = dynamic_positions[i][1][0]
                    y2 = dynamic_positions[i][1][1]
                    x1 = dynamic_positions[i][0][0]
                    x2 = dynamic_positions[i][0][1]
                    ax[i].set_xlim(0, self.xmax)
                    ax[i].set_ylim(y2, y1)
                    if not ax[i].yaxis_inverted():
                        ax[i].invert_yaxis()
                ax_ = ax[-1]
                
                
                # ax_.set_xlim(xmin, xmax)
                # ax.set_ylim(xmin, xmax)

            else:
                if ax is None:
                    plt.figure(figsize=(16, 9))
                    ax = plt.subplot(111)
                for i, i_bin in enumerate(self[which]):
                    coordinates = i_bin['coordinates']
                    stds = i_bin['stds']
                    slopes = i_bin['slopes']
                    ax = plotter.plot_coordinate_based(coordinates,
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
                        gridspec_kw={'hspace':2e-2})
                
                elif isinstance(ax, list):
                    assert len(ax) >= n_bins, 'Number of Axes provided ' \
                        "smaller than the number of bins"
                if n_bins == 1: ax=[ax]
                
                bins = self[which]
                for i, seg in enumerate(bins):
                    y1 = dynamic_positions[i][1][0]
                    y2 = dynamic_positions[i][1][1]
                    x1 = dynamic_positions[i][0][0]
                    x2 = dynamic_positions[i][0][1]
                    # ax[i].set_facecolor('black')
                    ax[i].imshow(seg, cmap=cmap, extent=(x1, x2, y1, y2))
                    # ax[i].set_xlim(x1, x2)
                    # ax[i].set_ylim(y1, y2)
                    ax[i].xaxis.set_visible(False)
                    ax[i].yaxis.set_visible(False)
                    # ax[i].imshow(seg, cmap=cmap)
                ax_ = ax[-1]
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
                    ax_ = ax
                else:
                    ax = self.plot('image', ax=ax, cmap=cmap)
                    ax_ = ax
        else:
            if ax is None:
                figsize = plotter.get_figure_size(self.metadata['dpi'],
                                                  self.shape[:2], 
                                                  zoom)
                plt.figure(figsize = figsize)
                ax = plt.subplot(111)
            ax.imshow(self[which], cmap=cmap)
            ax_ = ax
            ax.set_xlim(0, self[which].shape[1])
        if 'coordinate_based' not in which:
            ax_.xaxis.set_visible(False)
            ax_.yaxis.set_visible(False)
        if savefig is not None:
            plt.savefig(savefig)
            return ax
        if show:
            plt.show()
        else:
            return ax

    @abstractmethod
    def load_dict(self):
        """
        Abstract method for loading a dictionary.

        This method should be implemented by any non-abstract subclass of 
        Analyzer. The implementation should handle the loading of some kind 
        of dictionary data specific to that subclass.
        
        Returns
        -------
        Typically, this method would return the loaded dictionary, but the 
        exact return type and value will depend on the specific implementation 
        in the subclass.
        """
        pass

    @abstractmethod
    def from_dict(self):
        """
        Abstract method for setting the state of an object from a dictionary.

        This method should be implemented by any non-abstract subclass of 
        Analyzer. The implementation should set the state of an object based 
        on data provided in a dictionary.

        Parameters
        ----------
        This will vary depending on the subclass implementation, but typically 
        this method would accept a single argument: the dictionary containing 
        the data to use when setting the object's state.

        Returns
        -------
        Typically, this method would not return a value, but this will depend 
        on the specific implementation in the subclass.
        """
        pass

    @classmethod
    def from_buffer(cls, 
                    buffer: bytes, 
                    metadata: dict,
                    ext: str='.png'):
        """
        Receives an io byte buffer with the corresponding metadata and 
        creates an instance of the class. This class method is helpful in 
        situations where you have raw image data along with associated metadata 
        and need to create an Analyzer object.

        Parameters
        ----------
        buffer : bytes
            A buffer containing raw image data, typically in the form of bytes.
        metadata : dict
            A dictionary containing metadata related to the image. The specific 
            contents will depend on your application, but might include things 
            like the image's origin, resolution, or creation date.
        ext : str, optional
            The file extension of the image being loaded. Used to determine 
            the decoding method. Default is '.png'. If the 'ext' key is present 
            in the metadata dict, it will override this parameter.

        Returns
        -------
        An instance of the Analyzer class, initialized with the image and 
        metadata provided.
        """
        if 'ext' in metadata:
            ext = metadata['ext']
        if ext in IMAGE_EXTENSIONS:
            image = cv2.imdecode(np.frombuffer(buffer, np.uint8), -1)
            return cls.from_dict(image, metadata)
    
    def to_buffer(self, ext: str = '.png') -> bytes:
        """
        Converts the current instance of the Analyzer class to a byte buffer,
        which can be useful for serialization or for writing to a file. This
        method supports various image formats determined by the extension provided.

        Parameters
        ----------
        ext : str, optional
            The file extension for the output buffer. This will determine the 
            format of the output image. Default is '.png'. This method supports 
            any image format that is recognized by the OpenCV library.

        Returns
        -------
        bytes
            A byte string representing the image data in the format specified 
            by 'ext'. This can be directly written to a file or transmitted 
            over a network, among other things.

        Raises
        ------
        ValueError
            If the provided extension is not supported, a ValueError will be raised.
        """
        if ext in IMAGE_EXTENSIONS:
            _, buffer = cv2.imencode(ext, self.image)
            output = io.BytesIO(buffer)
        else:
            raise ValueError("Extension not supported")
        return output.getvalue()
    
    @property
    def shape(self) -> tuple:
        """
        A property that provides the shape of the image contained in the Analyzer instance.

        Returns
        -------
        tuple
            A tuple representing the shape of the image. For grayscale images, 
            this will be a 2-tuple (height, width). For color images, this 
            will be a 3-tuple (height, width, channels), where 'channels' 
            is typically 3 for an RGB image or 4 for an RGBA image.
        """
        return self.image.shape
    
    def __contains__(self, x):
        return x in self.values

    def __getitem__(self, x):
        return self.values.__getitem__(x)

    def __iter__(self):
        return self.values.__iter__()

    def __len__(self):
        return self.values.__len__()

    def __repr__(self) -> str:
        """
        A string representation method for the Analyzer class.

        This method plots the boundary of the image and prints the metadata 
        of the image.

        Returns
        -------
        str
            A string representing the metadata of the image.
        """
        self.plot(which='boundary', show=True)
        ret = self.metadata.__str__()
        return ret