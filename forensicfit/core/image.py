# -*- coding: utf-8 -*-
"""
used PyChemia code output class as a guide  
pychemia/code/codes.py
"""

__all__ = []
__version__ = '1.0'
__author__ = 'Pedram Tavadze'

# externals
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
        self.metadata = Metadata({'mode': 'image'})
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
            return cls(image, metadata)

    
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

    def flip_h(self):
        self.image = image_tools.flip(self.image)
        self.values['image'] = self.image
        self.metadata['flip_h'] = True

    def flip_v(self):
        self.image = np.fliplr(self.image)
        self.values['image'] = self.image
        self.metadata['flip_v'] = True

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

