# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
import pandas as pd
from .core import Tape, TapeAnalyzer
from .database import Database

def exists(db, filename,side='R',flip_h=False,analysis_mode='coordinate_based'):
    
    return db.gridfs_analysis.exists({
                "$and":[{"filename":filename},
                        {'metadata.side':side},
                        {'metadata.image.flip_h':flip_h},
                        {'metadata.analysis_mode':analysis_mode}]
                })

def analyze(db, tape,modes, iside, iflip, split_position, mask_threshold, gaussian_blur, ndivision, auto_crop, calculate_tilt, x_trim_param, verbose):
    tape.split_vertical(pixel_index=split_position, pick_side=iside)
    quality = tape.filename.plit("_")[0]
    if len(quality) == 4:
        if quality[-2:] == "HT":
            separation_method = "handtorn"
        elif quality[-2:] == "SC":
            separation_method = "cut"
    else:
        separation_method = "handtorn"
    quality = quality[0]
    streched = False
    if iflip:
     	tape.flip_h()
    analyed_tape = TapeAnalyzer(
            tape, mask_threshold, gaussian_blur, ndivision, auto_crop, calculate_tilt, verbose)
    analyed_tape.add_metadata("quality", quality)
    analyed_tape.add_metadata(
        "separation_method", separation_method)
    analyed_tape.add_metadata("streched", streched)
    # analyed_tape.add_metadata("side", side)
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

    
def from_excel(excel_file,
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
        weft_based_size=(600, 300),
        big_picture_size=(1200, 300),
        max_contrast_size=(4800, 300),
        split=True,
        side='both',
        auto_rotate=False,
        auto_crop=True,
        gaussian_blur=(15, 15),
        mask_threshold=60,
        split_position=0.5,
        calculate_tilt=True,
        verbose=False,
        skip=False,
        overwrite=False,
        db_name='forensicfit',
        host='localhost',
        port=27017,
        username="",
        password="",
        ignore_errors=False):
    
    db = Database(db_name, host, port, username, password)
    
    df = pd.read_excel(excel_file)
    
    for ientry in tqdm(range(len(df))):
        # Tape 1 Front
        name_f1 = df.iloc[ientry]['tape_f1']
        side_f1 = df.iloc[ientry]['side_f1']
        # Tape 2 Front
        all_exits = True
        for imode in modes:
            if not exists(db,name_f1,side_f1,analysis_mode=imode):
                all_exists = False
        
        if not all_exists:
            quality = ifile.split("_")[0]
            if len(quality) == 4:
                if quality[-2:] == "HT":
                    separation_method = "handtorn"
                elif quality[-2:] == "SC":
                    separation_method = "cut"
            else:
                separation_method = "handtorn"
            quality = quality[0]
            streched = False
            
        name_f2 = df.iloc[ientry]['tape_f2']
        side_f2 = df.iloc[ientry]['side_f2']
        flip_f = df.iloc[ientry]['flip_f']
        # Tape 1 Back
        
        
        name_b1 = df.iloc[ientry]['tape_b1']
        side_b1 = df.iloc[ientry]['side_b1']
        # Tape 2 Back
        name_b2 = df.iloc[ientry]['tape_b2']
        side_b2 = df.iloc[ientry]['side_b2']
        flip_b = df.iloc[ientry]['flip_b']
        
        
    # files = os.listdir(dir_path)
    
    
    output_dictionary = {}
    if 'coordinate_based' in modes:
        output_dictionary["coordinate_base"] = {}
    if'weft_based' in modes:
        output_dictionary["weft_based"] = {}
    if 'big_picture' in modes:
        output_dictionary["big_picture"] = {}
    if 'max_contrast' in modes:
        output_dictionary["max_contrast"] = {}
    c = 1
    for count in tqdm(range(nfiles), ascii=True, desc="Analyzing"):
        ifile = files[count]
        
        if ifile.split('.')[1] not in ['tif', 'jpg', 'bmp', 'png']:
            continue
        if db.exists_analysis(ifile) and skip:
            continue

        quality = ifile.split("_")[0]
        if len(quality) == 4:
            if quality[-2:] == "HT":
                separation_method = "handtorn"
            elif quality[-2:] == "SC":
                separation_method = "cut"
        else:
            separation_method = "handtorn"
        quality = quality[0]
        streched = False
        # side = 'Unknown'

        for iside in side:
            for iflip in [True, False]:

                tape = Tape(ifile, label=ifile)

                if split:
                    tape.split_vertical(pixel_index=split_position, pick_side=iside)
                if iflip:
                 	tape.flip_h()
                if ignore_errors:
                    try:
                        analyed_tape = TapeAnalyzer(
                            tape, mask_threshold, gaussian_blur, ndivision, auto_crop, calculate_tilt, verbose)
                    except:
                        print("Could not analyze file : %s" % ifile)
                else:
                    analyed_tape = TapeAnalyzer(
                        tape, mask_threshold, gaussian_blur, ndivision, auto_crop, calculate_tilt, verbose)
                analyed_tape.add_metadata("quality", quality)
                analyed_tape.add_metadata(
                    "separation_method", separation_method)
                analyed_tape.add_metadata("streched", streched)
                # analyed_tape.add_metadata("side", side)
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
