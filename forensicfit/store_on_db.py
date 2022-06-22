# -*- coding: utf-8 -*-

import os
from typing import Any, List, Dict
from pathlib import Path
import json
import multiprocessing
from multiprocessing import Pool, Process, cpu_count, current_process, freeze_support
import sys
import inspect
from unicodedata import name

from numpy import insert
from .core import Tape
from .database import Database
 
def get_chunks(files: List[Dict], n_processors: int):
    ret = [ [] for x in range(n_processors)]
    n_files = len(files)
    for i, file in enumerate(files):
        ret[i % n_processors].append(files[file])
    return ret

def worker(args):
    files = args['files'] 
    db_settings = args['db_settings']
    insert_options = args['insert_options']
    db = Database(**db_settings)
    for _, entry in enumerate(files):
        file_path = Path(entry['source'])
        if file_path.suffix not in ['.png', '.bmp', '.dib', '.jpeg', 
                                '.jpg', '.jpe', '.jp2', '.webp',
                                '.pbm', '.pgm', '.ppm', '.pxm', 
                                '.pnm', '.sr', '.ras', '.tiff',
                                '.tif', '.exr', '.hdr', '.pic']:
            continue
        tape = Tape.from_file(file_path)
        # print(f"{file_path.stem}")
        for key in entry:
            tape.metadata[key] = entry[key]
        db.insert(tape, **insert_options)
        
def get_files(path: Path, ret: list = [], ext: str ='.tif'):
    for x in path.iterdir():
        if x.is_file() and x.suffix == ext :
            ret.append({x.stem: {'source': x.as_posix(),
                                 'filename': x.stem}})
        elif x.is_dir():
            get_files(x, ret, ext)
    return ret

    

def store_on_db(
        dir_path='.',
        metadata_file = None,
        buffer_type = '.png',
        ext:str = '.tif', 
        overwrite: bool=False,
        skip: bool=True,
        db_name: str='forensicfit',
        host:str ='localhost',
        port: int=27017,
        username: str="",
        password:str ="",
        n_processors: int=1):
    """

    Parameters
    ----------
    dir_path : TYPE, optional
        DESCRIPTION. The default is '.'.
    dynamic_window : TYPE, optional
        DESCRIPTION. The default is True.
    verbose : TYPE, optional
        DESCRIPTION. The default is True.
    overwrite : TYPE, optional
        DESCRIPTION. The default is False.
    db_name : TYPE, optional
        DESCRIPTION. The default is 'forensicfit'.
    host : TYPE, optional
        DESCRIPTION. The default is 'localhost'.
    port : TYPE, optional
        DESCRIPTION. The default is 27017.
    username : TYPE, optional
        DESCRIPTION. The default is "".
    password : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    None.

    """
    if n_processors > cpu_count():
        n_processors = cpu_count()
        
    if metadata_file is not None:
        with open(metadata_file, 'r') as rf:
            metadata = json.load(rf)
    else:
        dir_path = Path(dir_path)
        
        metadata = get_files(dir_path, ext='.tif')
    chunks = get_chunks(metadata, n_processors)
    
    db_settings = dict(name=db_name,
                       host=host,
                       port=port,
                       username=username,
                       password=password)
    insert_options = dict(buffer_type=buffer_type,
                          overwrite=overwrite,
                          skip=skip)
    
    args = [{
            'files': x,
            'db_settings': db_settings,
            'insert_options': insert_options,
        } for x in chunks]
    with Pool(n_processors) as p:
        p.map(worker, iterable=args)

