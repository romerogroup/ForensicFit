# -*- coding: utf-8 -*-
"""
@author: Pedram Tavadze
used PyChemia code output class as a guide  
pychemia/code/codes.py
"""
import cv2
from matplotlib import pylab as plt
import numpy as np
from scipy import ndimage
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
import os
from ..utils import array_tools


class Material(Mapping):
    __metaclass__ = ABCMeta

    def __init__(self):
        """
        

        Returns
        -------
        None.

        """
        self.label = None
        self.image = None
        self.mode = 'material'
        self.material = None
        self.values = {}
        self.metadata = {'mode': 'material'}

    @property
    def is_loaded(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return not self.values == {}

    def read(self, filename):
        """
        

        Parameters
        ----------
        filename : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if not os.path.exists(filename):
            raise Exception("File %s does not exist" % filename)
        self.image = cv2.imread(filename, 0)
        
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

    def plot(self, savefig=None, cmap='gray', ax=None, rotate=0.0, show=False):
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
                
    def show(self, wait=0, savefig=None):
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
            
    def add_metadata(self, key, value):
        """
        

        Parameters
        ----------
        key : TYPE
            DESCRIPTION.
        value : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.metadata[key] = value

    @abstractmethod
    def load_dict(self):
        pass

    @abstractmethod
    def from_dict(self):
        pass

    def __contains__(self, x):
        return x in self.values

    def __getitem__(self, x):
        return self.values.__getitem__(x)

    def __iter__(self):
        return self.values.__iter__()

    def __len__(self):
        return self.values.__len__()

