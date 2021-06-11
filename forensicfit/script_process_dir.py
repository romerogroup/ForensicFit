# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import h5py
import cv2
from .core import Tape, TapeAnalyzer
from .database import Database


def process_directory(
        dir_path='.',
        modes=['coordinate_based', 'weft_based',
               'big_picture', 'max_contrast'],
        dynamic_window=True,
        nsegments={"h": 55, "m": 37, "l": 32},
        ndivision=6,
        window_tape=100,
        window_background=50,
        npoints=1000,
        x_trim_param=2,
        weft_based_size=(300, 30),
        big_picture_size=(1200, 30),
        max_contrast_size=(4800, 30),
        split=True,
        side='both',
        auto_rotate=False,
        auto_crop=True,
        gaussian_blur=(15, 15),
        mask_threshold=60,
        split_position=0.5,
        calculate_tilt=True,
        verbose=True,
        skip=False,
        overwrite=False,
        db_name='forensicfit',
        host='localhost',
        port=27017,
        username="",
        password="",
        ignore_errors=False):
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
    db = Database(db_name, host, port, username, password)
    if split:
        if side == 'both':
            side = ['L', 'R']
        else:
            side = [side]
    else:
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

    for count, ifile in enumerate(files):
        if ifile.split('.')[1] not in ['tif', 'jpg', 'bmp', 'png']:
            continue

        quality = ifile[0]

        for iside in side:
            for iflip in [True, False]:
                print(r"Analyzing file %s, side %s, flip %r, %d/%d" %
                      (ifile, iside, iflip, count+1, nfiles))
                tape = Tape(ifile, label=ifile)
                tape.add_metadata("quality", quality)
                if iflip:
                    if iside == 'R':
                        db.insert_item(tape, overwrite=overwrite, skip=skip)
                    tape.flip_h()
                tape.split_vertical(pixel_index=0.5)
                if ignore_errors:
                    try:
                        analyed_tape = TapeAnalyzer(
                            tape, mask_threshold, gaussian_blur, ndivision, auto_crop, calculate_tilt, verbose)
                    except:
                        print("++++++++++++++++++++++++++++")
                        print("Could not analyze file : %s" % ifile)
                else :
                    analyed_tape = TapeAnalyzer(
                        tape, mask_threshold, gaussian_blur, ndivision, auto_crop, calculate_tilt, verbose)
                analyed_tape.add_metadata("quality", quality)
                if 'coordinate_based' in modes:
                    analyed_tape.get_coordinate_based(npoints, x_trim_param)
                if 'weft_based' in modes:
                    analyed_tape.get_weft_based(window_background,
                                                window_tape,
                                                dynamic_window,
                                                weft_based_size,
                                                nsegments[quality.lower()])
                if 'big_picture' in modes:
                    analyed_tape.get_weft_based(window_tape,
                                                dynamic_window,
                                                weft_based_size,
                                                nsegments=4)

                if 'max_contrast' in modes:
                    analyed_tape.get_max_contrast(
                        window_background, window_tape, max_contrast_size)
                db.insert_item(analyed_tape)
