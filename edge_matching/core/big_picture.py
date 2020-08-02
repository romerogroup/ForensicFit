# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:41:08 2020

@author: Pedram tavadze
"""

__author__ = "Pedram Tavadze"

import cv2 
import numpy as np

class BigPicture:
    def __init__(self,
                 image,
                 nsegments,
                 boundary,
                 window_tape,
                 window_bg,
                 dynamic_window=False,
                 mask_threshold=60,
                 n_xsections=6,
                 nx_pixel=None,
                 ny_pixel=None,
                 ):
        
        self.image = image
        self.nsegments = nsegments
        self.boundary = boundary
        self.window_bg = window_bg
        self.window_tape = window_tape
        self.dynamic_window = dynamic_window
        self.mask_threshold = mask_threshold
        self.nx_pixel = nx_pixel 
        self.ny_pixel = ny_pixel # for each segment
        