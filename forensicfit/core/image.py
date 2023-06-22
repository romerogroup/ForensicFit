# -*- coding: utf-8 -*-
__all__ = []
__version__ = '1.0'
__author__ = 'Pedram Tavadze'

import io
import pathlib
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from pathlib import Path


import cv2
import numpy as np
import PIL
from matplotlib import pylab as plt
from matplotlib.axes import Axes
from scipy import ndimage
from skimage import exposure, filters
from typing import Union

from ..utils import copy_doc, image_tools, plotter
from .metadata import Metadata

IMAGE_EXTENSIONS = image_tools.IMAGE_EXTENSIONS
# PIL.Image.MAX_IMAGE_PIXELS = None
# mpl.rcParams['image.origin'] = 'lower'

class Image(Mapping):
    """
    A class used to represent an Image.

    This class is a Mapping that abstracts image data along with associated metadata. The primary data of an image
    is stored as a numpy array. Additional metadata is stored in a Metadata object, which can include any kind 
    of additional information relevant to the image, such as its mode or resolution.

    Parameters
    ----------
    image : np.ndarray
        The primary data of the image. This should be a 2D or 3D numpy array. For a 2D array, the dimensions 
        correspond to the height and width of the image. For a 3D array, the third dimension typically represents 
        the color channels of the image (e.g., red, green, and blue).

    kwargs : dict, optional
        Additional metadata for the image. This metadata is stored in a Metadata object. It can be accessed and 
        manipulated like a dictionary.

    Attributes
    ----------
    image : np.ndarray
        The primary data of the image.

    values : dict
        A dictionary that maps the string 'image' to the primary data of the image.

    metadata : Metadata
        An object of the Metadata class which stores the metadata of the image.

    See Also
    --------
    Metadata : MutableMapping used to store metadata for an image.
    """
    __metaclass__ = ABCMeta

    def __init__(self, image: np.ndarray, **kwargs):
        self.image = image
        self.values = {'image': self.image}
        self.metadata = Metadata({'mode': 'image', 
                                  'resolution': self.image.shape})
        self.metadata.update(kwargs)

    @classmethod
    def from_file(cls, filepath: Union[str, pathlib.Path]):
        """
        Create an Image object from a given file path.

        This class method reads an image file from a given path using the OpenCV library. It also extracts additional 
        image information using the PIL library. The extracted image and information are used to instantiate an Image 
        object.

        Parameters
        ----------
        filepath : Union[str, pathlib.Path]
            The path to the image file. This can be either a string or a pathlib.Path object.

        Returns
        -------
        Image
            An instance of the Image class, initialized with the image data and metadata from the specified file.

        Raises
        ------
        Exception
            If the file at the given path does not exist.
        """
        path = Path(filepath)
        if path.exists():
            image = cv2.imread(path.as_posix())
            pillow_image = PIL.Image.open(path)
            image_info = pillow_image.info
            pillow_image.close()
            if 'icc_profile' in image_info:
                del image_info['icc_profile']
            return cls(image, path=path, filename=path.name, **image_info)
        else:
            raise Exception(f"File {path.as_posix()} does not exist")
        
    @classmethod
    def from_buffer(cls, 
                    buffer: bytes, 
                    metadata: dict,
                    ext: str='.npz', 
                    allow_pickle: bool=False):
        """
        Create an Image object from a given buffer and its corresponding metadata.

        This class method reads an image from a given buffer using either numpy's load function or OpenCV's 
        imdecode, depending on the provided extension. The resulting image and the provided metadata are used to 
        instantiate an Image object.

        Parameters
        ----------
        buffer : io.BytesIO
            The buffer containing the image data.
        metadata : dict
            A dictionary containing metadata related to the image.
        ext : str, optional
            The extension of the image format, by default '.npz'.
        allow_pickle : bool, optional
            Allow loading pickled object arrays stored in npy files. Reasons for disallowing pickles include 
            security, as loading pickled data can execute arbitrary code.

        Returns
        -------
        Image
            An instance of the Image class, initialized with the image data and metadata from the specified buffer.

        Raises
        ------
        Exception
            If the extension of the image format is neither '.npz' nor in the list of supported IMAGE_EXTENSIONS.
        """
        if 'ext' in metadata:
            ext = metadata['ext']
        if ext == '.npz':
            values = dict(np.load(
                io.BytesIO(buffer), 
                allow_pickle=allow_pickle))
            del metadata['ext']
            return cls.from_dict(values, metadata)
            # cls = Image(values['image'], label=metadata['filename'])
            # cls.metadata = metadata
        elif ext in IMAGE_EXTENSIONS:
            image = cv2.imdecode(np.frombuffer(buffer, np.uint8), -1)
            obj = cls(image)
            obj.metadata.update(metadata)
            return obj

    @classmethod
    def from_dict(cls, values: dict, metadata: dict):
        """
        Create an Image object from a dictionary of values and metadata.

        This class method reads image data and corresponding metadata from two dictionaries and uses them 
        to instantiate an Image object. The 'image' key from the values dictionary is expected to hold the image data.

        Parameters
        ----------
        values : dict
            A dictionary containing the image data, expected to have an 'image' key.
        metadata : dict
            A dictionary containing metadata related to the image.

        Returns
        -------
        Image
            An instance of the Image class, initialized with the image data and metadata from the specified dictionaries.

        Raises
        ------
        KeyError
            If 'image' key is not found in the values dictionary.
        """
        return cls(values['image'], metadata)

    def to_buffer(self, ext: str = '.png') -> bytes:
        """
        Converts the Image instance into a bytes object.

        This method supports both 'npz' and other image formats. If 'ext' is '.npz', 
        it uses numpy's savez function to write the image to a BytesIO object. 
        For other image extensions, it uses OpenCV's imencode function.

        Parameters
        ----------
        ext : str, optional
            The file extension for the output buffer. Supported formats are '.npz' and the extensions 
            defined in IMAGE_EXTENSIONS. Default is '.png'.

        Returns
        -------
        bytes
            A bytes object representing the image.

        Raises
        ------
        ValueError
            If the extension specified is not supported.
        """
        if ext == '.npz':
            output = io.BytesIO()
            np.savez(output, self.values)
        elif ext in IMAGE_EXTENSIONS:
            is_success, buffer = cv2.imencode(ext, self.image)
            output = io.BytesIO(buffer)
        return output.getvalue()

    def to_file(self, filepath: str):
        """
        Writes the Image instance to a file.

        This method uses the imwrite function from the image_tools module to write the image data 
        to a file.

        Parameters
        ----------
        filepath : str
            The path to the file where the image should be written. 

        Returns
        -------
        None
        """
        image_tools.imwrite(filepath, self.image)
        return 
    
    @copy_doc(image_tools.exposure_control)
    def exposure_control(self, mode:str='equalize_adapthist', **kwargs):
        self.image = image_tools.exposure_control(self.image)
        self.metadata['exposure_control'] = mode
        if len(kwargs) != 0:
            for key in kwargs:
                self.metadata[key] = kwargs[key]
        return

    @copy_doc(image_tools.apply_filter)
    def apply_filter(self, mode:str, **kwargs):
        self.image = image_tools.apply_filter(self.image, **kwargs)
        self.metadata['filter'] = mode
        if len(kwargs) != 0:
            for key in kwargs:
                self.metadata[key] = kwargs[key]
        return

    def isolate(self,
                x_start: int = None, 
                x_end: int = None, 
                y_start: int = None, 
                y_end: int = None) -> np.ndarray:
        """
        Isolates a rectangular section from the image.

        This method slices a rectangular section from the image using the provided 
        x and y coordinates. If no coordinates are provided, it will return the full image.

        Parameters
        ----------
        x_start : int, optional
            The x-coordinate for the left boundary of the section. If not provided, it defaults to 0.
        x_end : int, optional
            The x-coordinate for the right boundary of the section. If not provided, it defaults to the image's width.
        y_start : int, optional
            The y-coordinate for the top boundary of the section. If not provided, it defaults to 0.
        y_end : int, optional
            The y-coordinate for the bottom boundary of the section. If not provided, it defaults to the image's height.

        Returns
        -------
        np.ndarray
            The sliced rectangular section from the image as a numpy array.
        """
        x_start = x_start or 0
        x_end = x_end or self.image.shape[1]
        y_start = y_start or 0
        y_end = y_end or self.image.shape[0]
        return self.image[y_start:y_end, x_start:x_end]
    
    def crop(self,
             x_start: int, 
             x_end: int, 
             y_start: int, 
             y_end: int) -> np.ndarray:
        """
        Crops the image using the specified x and y coordinates.

        This method uses the provided coordinates to create a rectangular crop of 
        the image. After cropping, it updates the image attribute and the resolution 
        in the metadata.

        Parameters
        ----------
        x_start : int
            The x-coordinate for the left boundary of the crop.
        x_end : int
            The x-coordinate for the right boundary of the crop.
        y_start : int
            The y-coordinate for the top boundary of the crop.
        y_end : int
            The y-coordinate for the bottom boundary of the crop.

        Returns
        -------
        None
        """
        self.image = self.isolate(**locals())
        self.metadata['resolution'] = self.image.shape
        return
    
    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the image.

        This property provides a convenient way to access the shape 
        (dimensions) of the image in the form of a tuple. The dimensions 
        are ordered as (height, width, channels), following the standard 
        convention of numpy arrays.

        Returns
        -------
        tuple
            A tuple containing the dimensions of the image.
        """
        return self.image.shape
        
    def convert_to_gray(self):
        """
        Converts the image to grayscale.

        This method uses the `to_gray` function from the `image_tools` module to 
        convert the original image to grayscale. This process involves 
        transforming the image from its original color space (typically RGB) to 
        grayscale. 

        The grayscale image only has one channel, reducing the dimensionality 
        and complexity of the image. This is useful in many image processing tasks 
        where color information is not necessary. After the conversion, the image 
        shape in the metadata is updated to reflect the change in channels.

        Note: This method modifies the image in-place.
        """
        self.image = image_tools.to_gray(self.image)
        self.metadata['resolution'] = self.image.shape
    
    def convert_to_rgb(self):
        """
        Converts the image to RGB color space.

        This method uses the `to_rgb` function from the `image_tools` module to 
        convert the image into the RGB (Red, Green, Blue) color space.

        The RGB color space represents images by specifying the intensity of 
        red, green, and blue channels. After the conversion, the image shape in 
        the metadata is updated to reflect the new three channels (height, width, RGB).

        Note: This method modifies the image in-place. If the image is already in 
        RGB color space, this operation won't change the image.

        """
        self.image = image_tools.to_rbg(self.image)
        self.metadata['resolution'] = self.image.shape


    def rotate(self, angle: float):
        """
        Rotates the image by a specified angle.

        This method uses the `rotate_image` function from the `image_tools` module to 
        rotate the image in-place by the specified angle in degrees. The rotation is 
        performed around the image center, not the origin.

        Parameters
        ----------
        angle : float
            The rotation angle in degrees. Positive values imply counter-clockwise 
            rotation and negative values imply clockwise rotation.

        Note: This method modifies the image in-place. 
        """
        self.image = image_tools.rotate_image(self.image, angle)
        
    def plot(self,
             savefig: str = None, 
             cmap:str='gray',
             ax: Axes = None,
             show: bool=False, 
             zoom: int=4,
             **kwargs) -> Axes:
        """
        Plots the current image.

        This method uses matplotlib to display the image. 
        If an Axes object is provided, the image is displayed on it, otherwise a new 
        plot is created. 
        It also allows to save the figure in a file.

        Parameters
        ----------
        savefig : str, optional
            The path to save the figure. If None, the figure is not saved. 
            The default is None.
        cmap : str, optional
            The color map used to display the image. It should be a valid 
            matplotlib colormap. The default is 'gray'.
        ax : Axes, optional
            The axes on which to display the image. If None, a new figure and axes are created. 
            The default is None.
        show : bool, optional
            If True, displays the figure immediately. The default is False.
        zoom : int, optional
            The zoom level of the image. The default is 4.

        Returns
        -------
        Axes
            The axes on which the image is displayed.
        """
        if ax is None:
            
            figsize = plotter.get_figure_size(dpi=self.metadata.dpi, 
                                              image_shape=self.shape[:2],
                                              zoom=zoom)
            plt.figure(figsize=figsize)
            ax = plt.subplot(111)
        image = self.image
        ax.imshow(image, cmap=cmap)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
        if show:
            plt.show()

        return ax
                
    def show(self, wait=0, savefig = None):
        """
        

        Parameters
        ----------
        wait : TYPE, optional
            DESCRIPTION. The default is 0.
        savefig : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        cv2.imshow(self.label, self.image)
        cv2.waitKey(wait)
        cv2.destroyAllWindows()
        if savefig is not None:
            cv2.imwrite(savefig, self.image)

    def resize(self, size: tuple = None, dpi: tuple = None):
        """
        Resizes the image according to the specified dimensions or dpi.

        This method can resize the image either based on a new size (width, height) or a 
        new dpi (dots per inch). If both size and dpi are provided, the size takes precedence. 
        The method updates the image's metadata accordingly.

        Parameters
        ----------
        size : tuple of int, optional
            The desired dimensions for the resized image as (width, height). 
            If provided, the image is resized to these dimensions.
            The default is None.
        dpi : tuple of int, optional
            The desired dpi for the resized image as (dpi_x, dpi_y). 
            If provided and no size is specified, the image is resized based on this dpi.
            The default is None.

        Returns
        -------
        None
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

    def flip_h(self):
        """
        Flips the image horizontally.

        This method flips the image along the vertical axis, i.e., from left to right. 
        The 'flip_h' metadata flag is set to True after the operation.

        Returns
        -------
        None
        """
        self.image = image_tools.flip(self.image)
        self.values['image'] = self.image
        self.metadata['flip_h'] = True
        self.metadata['resolution'] = self.image.shape

    def flip_v(self):
        """
        Flips the image vertically.

        This method flips the image along the horizontal axis, i.e., from top to bottom. 
        The 'flip_v' metadata flag is set to True after the operation.

        Returns
        -------
        None
        """
        self.image = np.fliplr(self.image)
        self.values['image'] = self.image
        self.metadata['flip_v'] = True
        self.metadata['resolution'] = self.image.shape

    def copy(self):
        """
        Creates a copy of the current image instance.

        This method generates a new Image object that is a copy of the current one, 
        preserving the values and metadata of the current Image instance.

        Returns
        -------
        Image
            A new Image object that is a copy of the current instance.
        """
        return self.from_dict(self.values, self.metadata)

    def __contains__(self, x):
        return x in self.values

    def __getitem__(self, x):
        return self.values.__getitem__(x)

    def __iter__(self):
        return self.values.__iter__()

    def __len__(self) -> int:
        return self.values.__len__()

    def __repr__(self) -> str:
        """
        Represents the Image instance as a string and plots the image.

        This method uses matplotlib to plot the image. After plotting, 
        it returns the string representation of the image's metadata.

        Returns
        -------
        str
            A string representation of the image's metadata.
        """
        self.plot(show=True)
        ret = self.metadata.__str__()
        return ret
