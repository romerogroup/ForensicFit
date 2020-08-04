#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:55:45 2020

@author: Pedram Tavadze
"""

import edge_matching
import cv2
import matplotlib.pylab as plt

# tape = edge_matching.TapeImage('/home/petavazohi/Pictures/goku_arsenal.jpg',
#                                 gaussian_blur=(15,15),
#                                 split=False,mask_threshold=113)
# tape = edge_matching.TapeImage('/home/petavazohi/Pictures/29557897.jpeg',
#                                 gaussian_blur=(3,3),
#                                 split=False,mask_threshold=102)
tape = edge_matching.TapeImage('LQ_774.tif',
                                gaussian_blur=(15,15),
                                split=True,mask_threshold=60,split_side='L')
# print(tape.image_tilt)
# tape.rotate_image(tape.image_tilt)

plt.figure()
tape.show(cmap='gray')
# # tape.show_binarized(cmap='gray')
tape.plot_boundary(color='red')

# plt.figure()
tape.auto_crop_y()
# tape.show(cmap='gray')
# # tape.show_binarized(cmap='gray')
# tape.plot_boundary(color='red')


