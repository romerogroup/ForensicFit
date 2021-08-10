# -*- coding: utf-8 -*-

import os
import tqdm 
import multiprocessing
from multiprocessing import Pool, Process, cpu_count, current_process, freeze_support
import sys
import inspect
from .core import Tape
from .database import Database
 



def chunks(files, nprocessors, db_name, host, port, username, password, overwrite, skip):
    ret = []
    nfiles = len(files)
    nchucks = nfiles//nprocessors
    start = 0
    end = 0
    for i in range(nprocessors):
        end = start + nchucks
        ret.append([files[start:end], db_name, host, port,
                    username, password, overwrite, skip, i])
        start = end
    if end != nfiles-1:
        for i in range(nfiles-end):
            ret[i][0].append(files[end+i])
    return ret

def init_child(lock):
    """
    Provide tqdm with the lock from the parent app.
    This is necessary on Windows to avoid racing conditions.
    """
    tqdm.tqdm.set_lock(lock)

def worker(args):
    files = args[0]
    db_name = args[1]
    host = args[2]
    port = args[3]
    username = args[4]
    password = args[5]
    overwrite = args[6]
    skip = args[7]
    pos = args[8]
    db = Database(db_name, host, port, username, password)

    nfiles = len(files)
    
    # pbar = tqdm.tqdm(total=nfiles, position=pos, desc="storing using proccess %d"%pos, leave=True)
    for count in tqdm.tqdm(range(nfiles), position=pos, desc="storing using process %d"%pos, leave=True):
        ifile = files[count]
        if ifile.split('.')[1] not in ['tif', 'jpg', 'bmp', 'png']:
            continue
        tape = Tape(ifile, label=ifile)
        quality = ifile.split("_")[0]
        if len(quality) == 4:
            if quality[-2:] == "HT":
                separation_method = "handtorn"
            elif quality[-2:] == "SC":
                separation_method = "cut"
        else:
            separation_method = "handtorn"
        streched = False
        side = 'Unknown'
        tape.add_metadata("quality", quality)
        tape.add_metadata("separation_method", separation_method)
        tape.add_metadata("streched", streched)
        tape.add_metadata("side", side)
        db.insert(tape, overwrite, skip)

        

        
    


def store_on_db(
        dir_path='.',
        verbose=True,
        overwrite=False,
        skip=True,
        db_name='forensicfit',
        host='localhost',
        port=27017,
        username="",
        password="",
        nprocessors=1):
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
    
    files = os.listdir(dir_path)
    cwd = os.getcwd()
    os.chdir(dir_path)
    if nprocessors == 1:
        worker(chunks(files, 1,  db_name,
                              host, port, username, password, overwrite, skip)[0])
    elif nprocessors > 1:
        freeze_support()
        
        lock = multiprocessing.Lock()
        if nprocessors > cpu_count():
            nprocessors = cpu_count()
        p = Pool(nprocessors, initializer=init_child, initargs=(lock,))
        args = chunks(files, nprocessors,  db_name,
                                            host, port, username, password, overwrite, skip)
        p.map(worker, args)

        p.close()
        p.join()
    os.chdir(cwd)
