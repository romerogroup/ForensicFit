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
from typing import List, Dict, Union, Any

import forensicfit as ff

import json
import numpy as np
import pandas as pd
import sys
if ff.HAS_PYMONGO:
    import gridfs
    import pymongo
    from bson.objectid import ObjectId
    from gridfs.grid_file import GridOut


def get_item(name: str, 
             lookup: Any,
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
            return tape_analyzer
        else:
            criteria = {'filename':fname}
            n_docs = db.count_documents(filter=criteria, collection='material')
            if n_docs == 0:
                return
            else:
                criteria = {"$and":[
                    {'filename':fname}, 
                    {'metadata.modified':True}]
                            }
                if db.count_documents(filter=criteria, collection='material') == 0 :
                    criteria = {'filename':fname}
            tape = db.find_one(filter=criteria, collection='material')
    elif isinstance(lookup, dict):
        key = fname
        if fname+'_mod' in lookup:
            key += '_mod'
        if key in lookup:
            path = Path(lookup[key]["source"])
            if path.exists():
                tape = ff.core.Tape.from_file(path)
            else:
                return
        else:
            return
    tape.split_v(side=side)
    tape.resize(dpi=tuple(args.dpi))
    tape_analyzer = ff.core.TapeAnalyzer(tape,
                                        mask_threshold=args.mask_threshold,
                                        gaussian_blur=args.gaussian_blur,
                                        n_divisions=args.n_divisions,
                                        auto_crop=args.auto_crop,
                                        correct_tilt=args.correct_tilt,
                                        padding=args.padding,
                                        )
    tape_analyzer.get_bin_based(window_background=args.window_background,
                                window_tape=args.window_tape,
                                dynamic_window=args.dynamic_window,
                                n_bins=args.n_bins, overlap=args.overlap,
                                border=args.border)
    if args.coordinate_based:
        tape_analyzer.get_coordinate_based(n_points=args.n_points)
    if args.exposure_control is not None:
        tape_analyzer.exposure_control(mode=args.exposure_control)
    if args.apply_filter is not None:
        tape_analyzer.apply_filter(mode=args.apply_filter)
    return tape_analyzer

def preprocess(entry: dict, 
               lookup: Any,
               args: argparse.Namespace):
    rotation_map = {'Backing' : 'Rotation?', 'Scrim' : 'Rotation?.1'}
    rot = ['', '-rotated']
    for tape in ['Tape 1', 'Tape 2']:
        for side in ['Backing', 'Scrim']:
            fname = entry[f'{tape} ({side})']
            tape_analyzer = get_item(fname, lookup, args)
            if tape_analyzer is None:
                continue
            rotated = 0
            if tape == 'Tape 1' and entry[rotation_map[side]]:
                tape_analyzer.flip_h()
                rotated = 1
            if args.bin_based:
                bins = tape_analyzer['bin_based']
                for ibin, b in enumerate(bins):
                    image = ff.core.Image(b)
                    if args.color:
                        image.convert_to_rgb()
                    else:
                        image.convert_to_gray()
                    if args.output_shape is not None:
                        image.resize(args.output_shape)
                    image.metadata.update(tape_analyzer.metadata.to_serial_dict)
                    image.metadata['bin'] = ibin
                    image.metadata['index'] = entry['idx']
                    image.metadata['filename'] = f"{fname}-{ibin}"
                    if args.cln_name is not None:
                        db = lookup
                        db.insert(image,
                                ext=args.ext,
                                collection=args.cln_name)
                    elif args.output is not None:
                        path = Path(args.output)
                        path /= f'{fname}{rot[rotated]}'
                        path /= f"{ibin}{args.ext}"
                        image.to_file(path)
            if args.max_contrast:
                bins = tape_analyzer['bin_based+max_contrast']
                for ibin, b in enumerate(bins):
                    image = ff.core.Image(b)
                    if args.color:
                        image.convert_to_rgb()
                    else:
                        image.convert_to_gray()
                    if args.output_shape is not None:
                        image.resize(args.output_shape)
                    image.metadata.update(tape_analyzer.metadata.to_serial_dict)
                    image.metadata['bin'] = ibin
                    image.metadata['index'] = entry['idx']
                    image.metadata['filename'] = f"{fname}-mxc-{ibin}"
                    if args.cln_name is not None:
                        db = lookup
                        db.insert(image,
                                ext='.png', 
                                collection=args.cln_name)
                    elif args.output is not None:
                        path = Path(args.output)
                        path /= f'{fname}{rot[rotated]}'
                        path /= f"{ibin}-mxc-{args.ext}"
                        image.to_file(path)
            if args.coordinate_based:
                bins = tape_analyzer['bin_based+coordinate_based']
                ret = {}
                if args.output is not None:
                    for ibin, b in enumerate(bins):
                        ret[f'bin_{ibin}'] = ff.utils.array_tools.serializer(b)
                    
                    path = Path(args.output)
                    path /= f'{fname}{rot[rotated]}'
                    path.mkdir(exist_ok=True)
                    path /= f"coordinate_based.json"
                    with open(path, 'w') as wf:
                        json.dump(ret, wf, indent=2)
    return

def get_chunks(entries: List[Dict], n_processors: int) -> List:
    ret = [ [] for x in range(n_processors)]
    n_entries = len(entries)
    for i, entry in enumerate(entries):
        ret[i % n_processors].append(entry)
    return ret

def worker(args: Dict[str, List]):
    entries = args['entries'] 
    args = args['parsed_args']
    if args.db_name is not None:
        lookup = ff.db.Database(name=args.db_name, 
                                host=args.db_host,
                                port=args.db_port)
    elif args.input is not None:
        with open(args.input, 'r') as rf:
            metadata = json.load(rf)
            lookup = {}
            for x in metadata:
                name = x["filename"]
                if x['modified']:
                    name += '_mod'
                lookup[name] = x
    if args.output is not None:
        path = Path(args.output)
        path.mkdir(exist_ok=True)
    for _, entry in enumerate(entries):
        preprocess(entry, lookup, parsed_args)
    if args.db_name is not None:
        lookup.disconnect()
        

if __name__ == '__main__':
    version = [int(x[:2]) for x in sys.version.replace(']','').replace('[','').split('.')][2]
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

    parser.add_argument('-np', '--n-processors', 
                        dest='n_processors',
                        type=int,
                        help='Number of processors available for this task.',
                        default=2)

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
                        default='.png',
                        help='The extension of file to be save.')
    parser.add_argument('-s', '--start',
                        dest='start',
                        type=int,
                        default=0,
                        help=('Index to start from in the excel file of the'
                              ' ground truth.'))
    parser.add_argument('-e', '--end',
                        dest='end',
                        type=int,
                        default=-1,
                        help=('Index to end to in the excel file of the'
                              ' ground truth.'))
    parser.add_argument('--excel-files',
                        dest='path_excel',
                        required=True,
                        nargs='+')

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
                        default=60,
                        type=int,
                        help=('Threshold for binarization of the image for edge'
                              ' detection.')
                        )
    parser.add_argument('--individual-entry', 
                        dest='individual_entry',
                        help=('files the individual entry from the excel files ' 
                              'and only processes those.'),
                        )
    parser.add_argument('--n-points',
                        dest='n_points',
                        type=int,
                        help='Number of points (in total) for coordinate based',
                        )

    parser.add_argument('--exposure-control',
                        dest='exposure_control',
                        default=None,
                        help=(
                            'Fixes the exposure of the image. options are: '
                            'equalize_hist and equalize_adapthist.'))
    parser.add_argument('--apply-filter',
                        dest='apply_filter',
                        default=None,
                        help=('Applies filter to the images. options are: '
                              'meijering, frangi, prewitt, sobel, scharr, '
                              'roberts, sato.'))
    parser.add_argument('--padding',
                        dest='padding',
                        default='tape',
                        help=('Padding in the y direction for the overlap.'
                              'black or tape'
                              ))
    parser.add_argument('--bin-based-border',
                        dest='border',
                        default='avg',
                        help=('The border used when applying window tape and '
                              'window background. avg or min.'
                              ))
    parser.add_argument('--dpi',
                        dest='dpi',
                        default=None,
                        type=int,
                        nargs=2,
                        help=('Output image, DPI.'))
    if version < 9:
        parser.add_argument('--max-contrast',
                            dest='max_contrast',
                            help=('To generate the maximum contrast for each bin'),
                            action=argparse.BooleanOptionalAction,
                            type=bool,
                            default=False)
        parser.add_argument('--coordinate-based',
                            dest='coordinate_based',
                            help=('To generate the coordinate based for each bin'),
                            action=argparse.BooleanOptionalAction,
                            type=bool,
                            default=False)
        parser.add_argument('--bin-based',
                            dest='bin_based',
                            help=('To generate the bin-based bins'),
                            action=argparse.BooleanOptionalAction,
                            type=bool,
                            default=True)
        parser.add_argument('--color', 
                            dest='color',
                            type=bool,
                            action=argparse.BooleanOptionalAction,
                            help='The output to be in gray scale or color.',
                            default=False,
                            )
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
        parser.add_argument('--dynamic-window', 
                            dest='dynamic_window',
                            type=bool,
                            default=True,
                            help='Use a dynamic window when scanning the edge',
                            action=argparse.BooleanOptionalAction,
                            )
    else:
        parser.add_argument('--max-contrast',
                            dest='max_contrast',
                            help=('To generate the maximum contrast for each bin'),
                            action='store_true')
        parser.add_argument('--no-max-contrast',
                            dest='max_contrast',
                            help=('To generate the maximum contrast for each bin'),
                            action='store_false')
        parser.add_argument('--coordinate-based',
                            dest='coordinate_based',
                            help=('To generate the coordinate based for each bin'),
                            action='store_true')
        parser.add_argument('--no-coordinate-based',
                            dest='coordinate_based',
                            help=('To generate the coordinate based for each bin'),
                            action='store_false')                            
        parser.add_argument('--bin-based',
                            dest='bin_based',
                            help=('To generate the bin-based bins'),
                            action='store_true')
        parser.add_argument('--no-bin-based',
                            dest='bin_based',
                            help=('To generate the bin-based bins'),
                            action='store_false')
        parser.add_argument('--color', 
                            dest='color',
                            help='The output to be in gray scale or color.',
                            action='store_true')
        parser.add_argument('--no-color', 
                            dest='color',
                            help='The output to be in gray scale or color.',
                            action='store_false')
        parser.add_argument('--auto-crop',
                            dest='auto_crop',
                            action='store_true',
                            help=('To crop the image according to the boundaries in '
                                'the y direction'))
        parser.add_argument('--no-auto-crop',
                            dest='auto_crop',
                            action='store_false',
                            help=('To crop the image according to the boundaries in '
                                'the y direction'))
        parser.add_argument('--correct-tilt',
                            dest='correct_tilt',
                            action='store_true',
                            help=('To calculate the angle of the image with the '
                                'horizontal line'))
        parser.add_argument('--no-correct-tilt',
                            dest='correct_tilt',
                            action='store_false',
                            help=('To calculate the angle of the image with the '
                                'horizontal line'))
        parser.add_argument('--dynamic-window', 
                            dest='dynamic_window',
                            help='Use a dynamic window when scanning the edge',
                            action='store_true',
                            )
        parser.add_argument('--no-dynamic-window', 
                            dest='dynamic_window',
                            help='Use a dynamic window when scanning the edge',
                            action='store_false',
                            )
        parser.set_defaults(max_contrast=False, 
                            coordinate_based=False,
                            bin_based=True,
                            color=False,
                            auto_crop=True,
                            correct_tilt=True,
                            dynamic_window=True)
    parsed_args = parser.parse_args()
    print('\n'.join(f'{k}={v}' for k, v in vars(parsed_args).items()))
    dfs = [pd.read_excel(x, engine='openpyxl') for x in parsed_args.path_excel]
    df = pd.concat(dfs)
    df = df[[x for x in df.columns if 'Tape' in x or 'Rotation' in x]]
    df['idx'] = np.arange(1, len(df) + 1 )
    df['Rotation?'] = df['Rotation?'].astype(bool)
    df['Rotation?.1'] = df['Rotation?.1'].astype(bool)
    if parsed_args.individual_entry is not None:
        dfs = []
        for col in df.columns:
            dfs.append(df[df[col] == parsed_args.individual_entry])
        df = pd.concat(dfs)
    else:
        df = df[parsed_args.start:parsed_args.end]
    del dfs
    chunks = get_chunks(df.to_dict('records'), parsed_args.n_processors)
    args = [
        {'entries': x,
        'parsed_args': parsed_args} 
        for x in chunks
        ]
    if parsed_args.n_processors == 1:
        worker(args[0])
    else:
        with Pool(parsed_args.n_processors) as p:
            p.map(worker, args)
    output = Path(parsed_args.output or '.')
    output /= f'log_{parsed_args.start}_{parsed_args.end}.json'
    with open(output, 'w') as wf:
        json.dump(parsed_args.__dict__, wf, indent=2)
