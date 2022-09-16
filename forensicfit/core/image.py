# -*- coding: utf-8 -*-
"""
used PyChemia code output class as a guide  
pychemia/code/codes.py
"""

__all__ = []
__version__ = '1.0'
__author__ = 'Pedram Tavadze'

import io
import pathlib
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from pathlib import Path

import PIL
import cv2
import numpy as np
import numpy.typing as npt
from matplotlib import pylab as plt
from scipy import ndimage
from skimage import exposure, filters

from ..utils import image_tools
from .metadata import Metadata

IMAGE_EXTENSIONS = image_tools.IMAGE_EXTENSIONS
PIL.Image.MAX_IMAGE_PIXELS = None

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
    def from_file(cls, filepath: str or pathlib.Path):

        path = Path(filepath)
        if path.exists():
            image = cv2.imread(path.as_posix())
            pillow_image = PIL.Image.open(path)
            image_info = pillow_image.info
            pillow_image.close()
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
    
    
    def exposure_control(self, mode:str='equalize_hist', **kwargs):
        """modifies the exposure

        Parameters
        ----------
        mode : str, optional
            Type of exposure correction. It can be selected from the options:
            ``'equalize_hist'`` or ``'equalize_adapthist'``. 
            `equalize_hist <https://scikit-image.org/docs/stable/api/skimage.exposure.html#equalize-hist>`_ 
            and `equalize_adapthist <https://scikit-image.org/docs/stable/api/skimage.exposure.html#equalize-adapthist>`
            use sk-image. by default 'equalize_hist'
        """
        exps = {'equalize_hist':exposure.equalize_hist,
                'equalize_adapthist':exposure.equalize_adapthist}
        assert mode in exps, 'Mode not valid.'
        self.image = exps[mode](self.image, **kwargs)
        self.metadata['exposure_control'] = mode
        if len(kwargs) != 0:
            for key in kwargs:
                self.metadata[key] = kwargs[key]
        return

    def apply_filter(self, mode:str, **kwargs):
        """Applies different types of filters to the image

        Parameters
        ----------
        mode : str
            Type of filter to be applied. The options are
            * ``'meijering'``: <Meijering neuriteness filter https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.meijering>_,
            * ``'frangi'``: < Frangi vesselness filter https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.frangi>_,
            * ``'prewitt'``: <Prewitt transform https://scikit-image.org/docs/stable/api/skimage.filters.html#prewitt>_,
            * ``'sobel'``: <Sobel filter https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel>_,
            * ``'scharr'``: <Scharr transform https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr>,
            * ``'roberts'``: <Roberts' Cross operator https://scikit-image.org/docs/stable/api/skimage.filters.html#examples-using-skimage-filters-roberts>_,
            * ``'sato'``: <Sato tubeness filter https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sato>_.
        """
        flts = {
            'meijering':filters.meijering,
            'frangi': filters.frangi,
            'prewitt': filters.prewitt,
            'sobel': filters.sobel,
            'scharr': filters.scharr,
            'roberts': filters.roberts,
            'sato': filters.sato
        }
        assert mode in flts, 'Filter not valid.'
        if mode == 'roberts':
            if self.image.ndim != 2:
                print('Cannot apply roberts to color images')
                return
        self.image = flts[mode](self.image, **kwargs)
        self.metadata['filter'] = mode
        if len(kwargs) != 0:
            for key in kwargs:
                self.metadata[key] = kwargs[key]
        return

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
    
    @property
    def shape(self):
        return self.image.shape
        
    def convert_to_gray(self):
        self.image = image_tools.to_gray(self.image)
        self.metadata['resolution'] = self.image.shape
    
    def convert_to_rgb(self):
        self.image = image_tools.to_rbg(self.image)
        self.metadata['resolution'] = self.image.shape
            
        
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

    def plot(self, savefig = None, cmap = 'gray', ax = None, rotate=0.0, show=False, **kwargs):
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

    def __len__(self):
        return self.values.__len__()

