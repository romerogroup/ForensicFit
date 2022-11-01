# -*- coding: utf-8 -*-
"""
used PyChemia code output class as a guide  
pychemia/code/codes.py
"""

__all__ = []
__version__ = '1.0'
__author__ = 'Pedram Tavadze'

from asyncore import dispatcher_with_send
import io
import pathlib
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from importlib.metadata import metadata
from pathlib import Path


import cv2
import numpy as np
import numpy.typing as npt
import PIL
from matplotlib import pylab as plt
from matplotlib.axes import Axes
from scipy import ndimage
from skimage import exposure, filters
from typing import Union

from ..utils import copy_doc, image_tools, plotter
from .metadata import Metadata

IMAGE_EXTENSIONS = image_tools.IMAGE_EXTENSIONS
PIL.Image.MAX_IMAGE_PIXELS = None
# mpl.rcParams['image.origin'] = 'lower'

class Image(Mapping):
    __metaclass__ = ABCMeta

    def __init__(self, image: npt.ArrayLike, **kwargs):
        """_summary_

        Parameters
        ----------
        image : np.array
            _description_
        label : str, optional
            _description_, by default None
        """        

        self.image = image
        self.values = {'image': self.image}
        self.metadata = Metadata({'mode': 'image', 
                                  'resolution': self.image.shape})
        self.metadata.update(kwargs)

    @classmethod
    def from_file(cls, filepath: Union[str, pathlib.Path]):

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
        """receives an io byte buffer with the corresponding metadata and creates 
        the image class

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
        return cls(values['image'], metadata)
        
                    
    def to_buffer(self, ext: str = '.png'):
        if ext == '.npz':
            output = io.BytesIO()
            np.savez(output, self.values)
        elif ext in IMAGE_EXTENSIONS:
            is_success, buffer = cv2.imencode(ext, self.image)
            output = io.BytesIO(buffer)
        return output.getvalue()

    def to_file(self, filepath: str):
        image_tools.imwrite(filepath, self.image)
        return 
    
    @copy_doc(image_tools.exposure_control)
    def exposure_control(self, mode:str='equalize_hist', **kwargs):
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
        """isolates a rectangle from the image

        Parameters
        ----------
        x_start : int
            _description_
        x_end : int
            _description_
        y_start : int
            _description_
        y_end : int
            _description_

        Returns
        -------
        np.ndarray
            _description_
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
        self.image = self.isolate(**locals())
        self.metadata['resolution'] = self.image.shape
        return
    
    @property
    def shape(self):
        return self.image.shape
        
    def convert_to_gray(self):
        self.image = image_tools.to_gray(self.image)
        self.metadata['resolution'] = self.image.shape
    
    def convert_to_rgb(self):
        self.image = image_tools.to_rbg(self.image)
        self.metadata['resolution'] = self.image.shape


    def rotate(angle: float):
        self.image = image_tools.rotate_image(self.image, angle)
        
    def plot(self,
             savefig: str = None, 
             cmap:str='gray',
             ax: Axes = None,
             show: bool=False, 
             zoom: int=4,
             **kwargs):
        """
        

        Parameters
        ----------
        savefig : TYPE, optional
            DESCRIPTION. The default is None.
        cmap : TYPE, optional
            DESCRIPTION. The default is 'gray'.
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        rotate : TYPE, optional
            DESCRIPTION. The default is 0.0.

        Returns
        -------
        None.

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
        if show:
            plt.show()
        # if savefig is not None:
        #     cv2.imwrite(savefig, self.image)
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
        self.image = image_tools.flip(self.image)
        self.values['image'] = self.image
        self.metadata['flip_h'] = True
        self.metadata['resolution'] = self.image.shape

    def flip_v(self):
        self.image = np.fliplr(self.image)
        self.values['image'] = self.image
        self.metadata['flip_v'] = True
        self.metadata['resolution'] = self.image.shape

    def copy(self):
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
        self.plot(show=True)
        ret = self.metadata.__str__()
        return ret
