#!/usr/bin/env python3

"""This module will contain a parallelizd script to generate bin based 
data for ML
"""

__author__ = 'Pedram Tavadze'
__date__ = "20220727"
__status__ = 'Production'

import argparse
from multiprocessing import Pool
from pathlib import Path
from typing import List, Dict, Union

from traitlets import default

import forensicfit as ff

import json
import numpy as np
import pandas as pd
import gridfs
import pymongo
from bson.objectid import ObjectId
from gridfs.grid_file import GridOut


def get_item(name: str, 
             lookup: Union[ff.db.Database, Dict],
             args: argparse.Namespace) -> ff.core.TapeAnalyzer: 
    """Gets a scan from the source database

    Parameters
    ----------
    name : str
        file name
    db : forensicfit.db.Database
        The mongodb object to get the file from

    Returns
    -------
    forensicfit.core.TapeAnalyzer
        analyzed tape scan as a forensicfit object
    """
    # name = name.decode("utf-8")
    fname = f"{name[:-2]}"
    side = name[-1]
    if isinstance(lookup, ff.db.Database):
        criteria = {"$and":[]}
        criteria['$and'].append({'metadata.cropped':args.auto_crop})
        criteria['$and'].append({'metadata.gaussian_blur':list(args.gaussian_blur)})
        criteria['$and'].append({'metadata.mask_threshold':args.mask_threshold})
        criteria['$and'].append({'metadata.n_divisions':args.n_divisions})
        criteria['$and'].append({'metadata.cropped':args.auto_crop})

        criteria['$and'].append(
            {'metadata.analysis.bin_based.dynamic_window':args.dynamic_window}
            )
        criteria['$and'].append({'metadata.analysis.bin_based.n_bins':args.n_bins})
        criteria['$and'].append(
            {'metadata.analysis.bin_based.window_background':args.window_background}
            )
        criteria['$and'].append(
            {'metadata.analysis.bin_based.window_tape':args.window_tape}
            )
        db = lookup
        criteria['$and'].append({'filename':fname})
        _id = db.exists(collection='analysis', filter=criteria)
        if _id:
            tape_analyzer = db.find_with_id(_id, 'analysis')
            if len(tape_analyzer['bin_based']) == 0:
                db.delete(collection='analysis', filter=criteria)
                tape_analyzer = get_item(name, db, args)
        else:
            criteria = {'filename':fname}
            n_docs = db.count_documents(filter=criteria, collection='material')
            if n_docs == 0:
                return
            else:
                criteria = {"$and":[{'filename':fname}, {'metadata.modified':True}]}
                if db.count_documents(filter=criteria, collection='material') == 0 :
                    criteria = {'filename':fname}
            tape = db.find_one(filter=criteria, collection='material')
            
    elif isinstance(lookup, dict):
        key = fname
        if fname+'_mod' in lookup:
            key += '_mod'
        tape = ff.core.tape.from_file(lookup[key]["source"])
    tape.split_v(side=side)
    tape_analyzer = ff.core.TapeAnalyzer(tape,
                                        mask_threshold=args.mask_threshold,
                                        gaussian_blur=args.gaussian_blur,
                                        n_divisions=args.n_divisions,
                                        auto_crop=args.auto_crop,
                                        calculate_tilt=args.calculate_tilt,
                                        )
    tape_analyzer.get_bin_based(window_background=args.window_background,
                                window_tape=args.window_tape,
                                dynamic_window=args.dynamic_window,
                                n_bins=args.n_bins, overlap=args.overlap)
        
    return tape_analyzer

def preprocess(entry: dict, 
               lookup: Union[ff.db.Database, Dict],
               args: argparse.Namespace):
    rotation_map = {'Backing' : 'Rotation?', 'Scrim' : 'Rotation?.1'}
    for tape in ['Tape 1', 'Tape 2']:
        for side in ['Backing', 'Scrim']:
            fname = entry[f'{tape} ({side})']
            tape_analyzer = get_item(fname, lookup, args)
            if tape_analyzer is None:
                print(fname)
                continue
            if tape == 'Tape 1' and entry[rotation_map[side]]:
                tape_analyzer.flip_h()
            # print(f"filename: {fname:15}, rotation: {tape_analyzer.metadata.flip_h}, index: {entry['idx']}")
            
            for ibin in range(args.n_bins):
                image = ff.core.Image(tape_analyzer['bin_based'][ibin])
                if args.color:
                    image.convert_to_rgb()
                else:
                    image.convert_to_gray()
                if 'outpt_shape' in args:
                    image.resize(args.output_shape)
                image.metadata.update(tape_analyzer.metadata.to_serial_dict)
                image.metadata['bin'] = ibin
                image.metadata['index'] = entry['idx']
                image.metadata['filename'] = f"{fname}_{ibin}"
                if 'cln_name' in args:
                    db = lookup
                    db.insert(image,
                            ext='.png', 
                            collection=args.cln_name)
                elif 'output' in args:
                    path = Path(args.output)
                    path /= f"{fname}_{ibin}_{args.ext}"
                    image.to_file(path)
    return

def get_chunks(entries: List[Dict], n_processors: int) -> List:
    ret = [ [] for x in range(n_processors)]
    n_entries = len(entries)
    for i, entry in enumerate(entries):
        ret[i % n_processors].append(entry)
    return ret

def worker(args: Dict[str, List]):
    entries = args['entries'] 
    parsed_args = args['parsed_args']
    if 'db_name' in parsed_args:
        lookup = ff.db.Database(name=args.db_name, 
                            host=args.db_host,
                            port=args.db_port)
    elif 'input' in parsed_args:
        with open(args.input, 'r') as rf:
            metadata = json.load(rf)
            lookup = {}
            for x in lookup:
                name = x["filename"]
                if x['modified']:
                    name += '_mod'
                lookup[name] = x
    for _, entry in enumerate(entries):
        preprocess(entry, lookup, parsed_args)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dbn', '--database-name', 
                           dest='db_name',
                           help='The name of the database.')
    parser.add_argument('-dbh', '--database-host', 
                           dest='db_host',
                           default='localhost',
                           help='The name of the host for the database server.')
    parser.add_argument('-dbp', '--database-port', 
                           dest='db_port',
                           type=int,
                           default=27017,
                           help='The port of the database server.')
    # TODO add username and password arguments for database.
    parser.add_argument('-cln', '--collection-name', 
                           dest='cln_name',
                           help=('The name of the target collection in which the'
                                 'data will be stored'))
    parser.add_argument('-i', '--input',
                                dest='input',
                                help=('Path to the metadata.json file that' 
                                      'contains path all the files. This file ' 
                                      'can be generated by create_metadata.py '
                                      'script')
                                )
    parser.add_argument('-o', '--output',
                        dest='output',
                        help=('Path to the directory for the output '
                              'image')
                        )
    parser.add_argument('--dynamic-window', 
                        dest='dynamic_window',
                        type=bool,
                        help='Use a dynamic window when scanning the edge',
                        action=argparse.BooleanOptionalAction,
                        )
    parser.add_argument('-np', '--n-processors', 
                        dest='n_processors',
                        type=int,
                        help='Number of processors available for this task.',
                        default=2)
    parser.add_argument('--auto-crop',
                        dest='auto_crop',
                        type=bool,
                        default=True,
                        action=argparse.BooleanOptionalAction,
                        help=('To crop the image according to the boundaries in '
                              'the y direction'))
    parser.add_argument('--correct-tilt',
                        dest='correct_tilt',
                        type=bool,
                        default=True,
                        action=argparse.BooleanOptionalAction,
                        help=('To calculate the angle of the image with the '
                              'horizontal line'))
    parser.add_argument('-nb', '--n-bins',
                        dest='n_bins',
                        type=int,
                        required=True,
                        help='Number of bins to divide the tape in.',
                        )
    parser.add_argument('-nd', '--n-divisions',
                        dest='n_divisions',
                        type=int,
                        default=6,
                        help=('Parameter to find the edge. Number of divisions '
                              'in the x direction'))
    parser.add_argument('-wbg', '--window-background',
                        dest='window_background',
                        type=int,
                        required=True,
                        help=('Number of pixels to include from the background '
                              '(black) of the image. This parameter is measured '
                              'from the edge of the tape towards the background.'
                              ))
    parser.add_argument('-wtp', '--window-tape',
                        dest='window_tape',
                        type=int,
                        required=True,
                        help=('Number of pixels to include from the tape '
                              'of the image. This parameter is measured '
                              'from the edge of the tape towards the tape.'))
    parser.add_argument('--overlap',
                        dest='overlap',
                        type=int,
                        default=100,
                        help=('Number of pixels to include from the adjacent '
                              'tape.'))
    parser.add_argument('-x', '-extension',
                        dest='ext',
                        help='The extension of file to be save.')
    parser.add_argument('-s', '--start',
                        dest='start',
                        type=int,
                        help=('Index to start from in the excel file of the'
                              ' ground truth.'))
    parser.add_argument('-e', '--end',
                        dest='end',
                        type=int,
                        help=('Index to end to in the excel file of the'
                              ' ground truth.'))
    parser.add_argument('--excel-files',
                        dest='path_excel',
                        required=True,
                        nargs='+')
    parser.add_argument('--color', 
                        dest='color',
                        type=bool,
                        action=argparse.BooleanOptionalAction,
                        help='The output to be in gray scale or color.'
                        )
    parser.add_argument('--output-shape', 
                        dest='output_shape',
                        type=int,
                        nargs=2,
                        help='The output to be in gray scale or color.'
                        )
    parser.add_argument('--gaussian-blur', 
                        dest='gaussian_blur',
                        type=int,
                        nargs=2,
                        help=('Gaussian blur applied to the image for edge' 
                              ' detection.'),
                        default=[15, 15],
                        )
    parser.add_argument('--mask-threshold', 
                        dest='mask_threshold',
                        type=int,
                        help=('Threshold for binarization of the image for edge'
                              ' detection.')
                        )
    parsed_args = parser.parse_args()
    
    dfs = [pd.read_excel(x, engine='openpyxl') for x in parsed_args.path_excel]
    df = pd.concat(dfs)
    df['idx'] = np.arange(1, len(df) + 1 )
    del dfs
    
    chunks = get_chunks(df.to_dict('records'), parsed_args.n_processors)
    args = [
        {'entries': x,
        'parsed_args': parsed_args} 
        for x in chunks
        ]
    print(args)