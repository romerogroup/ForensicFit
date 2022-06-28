# -*- coding: utf-8 -*-
"""
used PyChemia code output class as a guide  
pychemia/code/codes.py
"""

__all__ = []
__version__ = '1.0'
__author__ = 'Pedram Tavadze'

# externals
from re import X
import cv2
from matplotlib import pylab as plt
import numpy as np
import numpy.typing as npt
from scipy import ndimage
# built-ins
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from pathlib import Path
import io
# internals
from .metadata import Metadata
from ..utils import image_tools

IMAGE_EXTENSIONS = image_tools.IMAGE_EXTENSIONS 


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
    def from_file(cls, filepath: str):

        path = Path(filepath)
        if path.exists():
            image = cv2.imread(path.as_posix())
            return cls(image, path=path, filename=path.name)
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

    
    def isolate(self,
             x_start: int, 
             x_end: int, 
             y_start: int, 
             y_end: int) -> np.ndarray:
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
        return self.image[y_start:y_end, x_start:x_end]
    
    def crop(self,
             x_start: int, 
             x_end: int, 
             y_start: int, 
             y_end: int) -> np.ndarray:
        self.image = self.isolate(**locals())
        self.metadata['resolution'] = self.image.shape
        return
    
    @classmethod
    def from_dict(cls, values: dict, metadata: dict):
        return cls(values['image'], metadata)
        
        
    @property
    def shape(self):
        return self.image.shape
        
    def convert_to_gray(self):
        if len(self.shape) == 2:
            return 
        elif len(self.shape) == 3:
            if self.shape[2] == 1:
                return 
            elif self.shape[2] == 3:
                self.image = image_tools.to_gray(self.image)
                self.metadata['resolution'] = self.image.shape
    
    def convert_to_rgb(self):
        if len(self.shape) == 3 and self.shape[2] == 3:
            return
        else:
            self.image = image_tools.to_rbg(self.image)
            self.metadata['resolution'] = self.image.shape
            
                
    def to_buffer(self, ext: str = '.png'):
        if ext == '.npz':
            output = io.BytesIO()
            np.savez(output, self.values)
        elif ext in IMAGE_EXTENSIONS:
            is_success, buffer = cv2.imencode(ext, self.image)
            output = io.BytesIO(buffer)
        return output.getvalue()
        
    @property
    def aspect_ratio(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        gcd = np.gcd(self.image.shape[0], self.image.shape[1])
        return (self.image.shape[1]//gcd, self.image.shape[0]//gcd)

    def plot(self, savefig = None, cmap = 'gray', ax = None, rotate=0.0, show=False):
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
            plt.figure(figsize=(16, 9))
            ax = plt.subplot(111)
        image = self.image
        if rotate != 0.0:
            image = ndimage.rotate(image, rotate)
        ax.imshow(image, cmap=cmap)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        if show:
            plt.show()
        if savefig is not None:
            cv2.imwrite(savefig, self.image)
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

    def resize(self, size):
        self.image = image_tools.resize(self.image, size)
        self.values['image'] = self.image
        self.metadata['resize'] = size
        self.metadata['resolution'] = self.image.shape

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

    def __len__(self):
        return self.values.__len__()

