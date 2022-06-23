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
from .utils.image_tools import IMAGE_EXTENSIONS


def get_chunks(files: List[Dict], n_processors: int):
    ret = [ [] for x in range(n_processors)]
    n_files = len(files)
    for i, file in enumerate(files):
        ret[i % n_processors].append(files[i])
    return ret

def worker(args):
    files = args['files'] 
    db_settings = args['db_settings']
    insert_options = args['insert_options']
    db = Database(**db_settings)
    for _, entry in enumerate(files):
        file_path = Path(entry['source'])
        if not file_path.exists():
            print(f"{file_path.as_posix()} does not exist")
            return 
        if file_path.suffix not in IMAGE_EXTENSIONS:
            continue
        tape = Tape.from_file(file_path)
        # print(f"{file_path.stem}")
        for key in entry:
            tape.metadata[key] = entry[key]
        db.insert(tape, **insert_options)
        
def get_files(path: Path, ret: list = []):
    for x in path.iterdir():
        if x.is_file() and x.suffix in IMAGE_EXTENSIONS :
            ret.append({x.stem: {'source': x.as_posix(),
                                 'filename': x.stem}})
        elif x.is_dir():
            get_files(x, ret)
    return ret

    

def store_on_db(
        dir_path='.',
        metadata_file = None,
        ext: str = '.png',
        overwrite: bool=False,
        skip: bool=True,
        db_name: str='forensicfit',
        host:str ='localhost',
        port: int=27017,
        username: str="",
        password:str ="",
        n_processors: int=1):
    """_summary_

    Parameters
    ----------
    dir_path : str, optional
        _description_, by default '.'
    metadata_file : _type_, optional
        _description_, by default None
    ext : str, optional
        extension for storage on mongodb, by default '.png'
    overwrite : bool, optional
        _description_, by default False
    skip : bool, optional
        _description_, by default True
    db_name : str, optional
        _description_, by default 'forensicfit'
    host : str, optional
        _description_, by default 'localhost'
    port : int, optional
        _description_, by default 27017
    username : str, optional
        _description_, by default ""
    password : str, optional
        _description_, by default ""
    n_processors : int, optional
        _description_, by default 1
    """    
    if n_processors > cpu_count():
        n_processors = cpu_count()
        
    if metadata_file is not None:
        with open(metadata_file, 'r') as rf:
            metadata = json.load(rf)
    else:
        dir_path = Path(dir_path)
        
        metadata = get_files(dir_path)
    chunks = get_chunks(metadata, n_processors)
    
    db_settings = dict(name=db_name,
                       host=host,
                       port=port,
                       username=username,
                       password=password)
    insert_options = dict(ext=ext,
                          overwrite=overwrite,
                          skip=skip)
    
    args = [{
            'files': x,
            'db_settings': db_settings,
            'insert_options': insert_options,
        } for x in chunks]
    with Pool(n_processors) as p:
        p.map(worker, iterable=args)

