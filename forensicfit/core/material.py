# -*- coding: utf-8 -*-
"""
@author: Pedram Tavadze
used PyChemia code output class as a guide  
pychemia/code/codes.py
"""
__all__ = []
__version__ = '1.0'
__author__ = 'Pedram Tavadze'

from . import Image
import numpy.typing as npt
from abc import ABCMeta, abstractmethod

class Material(Image):

    def __init__(self, image: npt.ArrayLike,
                 **kwargs):
        """
        

        Returns
        -------
        None.

        """
        super().__init__(image, **kwargs)

        
        self.metadata['mode'] = 'material'
        self.metadata['material'] = None
        
