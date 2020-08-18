# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import h5py
import cv2
from .preprocess import TapeImage

def process_directory(dir_path='.',
                      output_format='json',
                      modes=['coordinate_based','weft_based','big_picture','max_contrast'],
                      dynamic_window=True,
                      nsegments=39,
                      window_tape=100,
                      window_background=50,
                      npoints=1000,
                      weft_based_size=(300,30),
                      big_picture_size=(1200,30),
                      max_contrast_size=(4800,30),
                      split=True,
                      side='both',
                      auto_rotate=False,
                      gaussian_blur=(15,15),
                      mask_threshold=60,
                      split_position=0.5,
                      plot=False):
    """
    

    Parameters
    ----------
    dir_path : TYPE, optional
        DESCRIPTION. The default is '.'.
    output_format : TYPE, optional
        DESCRIPTION. The default is 'json'.
    modes : TYPE, optional
        DESCRIPTION. The default is ['coordinate_based','weft_based','big_picture','max_contrast'].
    dynamic_window : TYPE, optional
        DESCRIPTION. The default is True.
    nsegments : TYPE, optional
        DESCRIPTION. The default is 39.
    window_tape : TYPE, optional
        DESCRIPTION. The default is 100.
    window_background : TYPE, optional
        DESCRIPTION. The default is 50.
    npoints : TYPE, optional
        DESCRIPTION. The default is 1000.
    split : TYPE, optional
        DESCRIPTION. The default is True.
    side : TYPE, optional
        DESCRIPTION. The default is 'both'.
    auto_rotate : TYPE, optional
        DESCRIPTION. The default is False.
    gaussian_blur : TYPE, optional
        DESCRIPTION. The default is (15,15).
    mask_threshold : TYPE, optional
        DESCRIPTION. The default is 60.
    split_position : TYPE, optional
        DESCRIPTION. The default is 0.5.
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    if split :
        if side == 'both':
            side=['L','R']
        else :
            side = [side]
    else :
        side = ['L']
    files = os.listdir(dir_path)
    nfiles = len(files)
    
    output_dictionary = {}
    if 'coordinate_based' in modes:
        output_dictionary["coordinate_base"] = {} 
    if'weft_based' in modes:
        output_dictionary["weft_based"] = {} 
    if 'big_picture' in modes:
        output_dictionary["big_picture"] = {}
    if 'max_contrast' in modes:
        output_dictionary["max_contrast"] = {}
        
    for ifile in files:
        for iside in side:
            tape_image = TapeImage(ifile,
                               tape_image=ifile+'_'+iside,
                               split=split,
                               side=iside,
                               gaussian_blur=gaussian_blur,
                               split_position=split_position,
                               mask_threshold=60)
            tape_image.auto_crop_y()
            if 'coordinate_based' in modes:
                coords = tape_image.coordinate_based(plot=plot,
                                                     x_trim_param=6,
                                                     npoints=npoints)
                output_dictionary["coordinate_base"][ifile+'_'+iside] = coords
            if'weft_based' in modes:
                segments = tape_image.weft_based(window_background=window_background,
                                                 window_tape=window_tape, 
                                                 dynamic_window=dynamic_window,
                                                 nsegments=nsegments,
                                                 size=weft_based_size,
                                                 plot=plot)
                output_dictionary["weft_based"][ifile+'_'+iside] = segments
            if 'big_picture' in modes:
                segments = tape_image.weft_based(window_background=window_background,
                                 window_tape=window_tape, 
                                 dynamic_window=dynamic_window,
                                 nsegments=4,
                                 plot=plot)
                output_dictionary["big_picture"][ifile+'_'+iside] = segments
            if 'max_contrast' in modes:
                max_contrast = tape_image.max_contrast(window_background=window_background,
                                                       window_tape=window_tape,
                                                       plot=plot)
                output_dictionary["max_contrast"][ifile+'_'+iside] = max_contrast
    
    if output_format == 'json' :
        wf = open('output.json','w')
        json.dump(output_dictionary,
                  wf,
                  sort_keys=True,
                  indent=4,
                  separators=(',', ': '))
        
    elif output_format == 'h5py':
        wf = h5py.File('output.h5py','w')
        if 'coordinate_based' in modes:
            coord_dset = wf.create_dataset("output/coordinate_base", 
                                           shape=(nfiles,npoints,2))
            for ilabel in output_dictionary["coordinate_base"]:
                
        if'weft_based' in modes:
            weft_dset = wf.create_dataset("output/weft_based", 
                                          shape=(nfiles,
                                                 weft_based_size[0],
                                                 weft_based_size[1]))         
        if 'big_picture' in modes:
            big_picture_dset = wf.create_dataset("output/big_picture",
                                                 shape=(nfiles,
                                                        big_picture_size[0],
                                                        big_picture_size[1]))
        if 'max_contrast' in modes:
            max_contrast_dset = wf.create_dataset("output/max_contrast", 
                                                  shape=(nfiles,
                                                         max_contrast_size[0],
                                                         max_contrast_size[1]))
        
    elif output_format == 'txt':
        if not os.path.exists('txt_outputs'):
            os.mkdir('txt_outputs')
    elif output_format == 'image':
        if not os.path.exists('image_outputs'):
            os.mkdir('image_outputs')
