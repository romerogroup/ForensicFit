# -*- coding: utf-8 -*-

import os
from multiprocessing import Pool
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
        if end >= nfiles:
            end = nfiles-1
        ret.append([files[start:end], db_name, host, port,
                    username, password, overwrite, skip])
        start = end
    return ret


def worker(args):
    files = args[0]
    db_name = args[1]
    host = args[2]
    port = args[3]
    username = args[4]
    password = args[5]
    overwrite = args[6]
    skip = args[7]
    db = Database(db_name, host, port, username, password)

    nfiles = len(files)

    for count, ifile in enumerate(files):
        if ifile.split('.')[1] not in ['tif', 'jpg', 'bmp', 'png']:
            continue
        print(count, ifile)
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
        db.insert_item(tape, overwrite, skip)


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
    worker(chunks(files, 1,  db_name,
                          host, port, username, password, overwrite, skip)[0])
    # p = Pool(nprocessors)
    # p.map(worker, chunks(files, nprocessors,  db_name,
    #                       host, port, username, password, overwrite, skip))
    # p.close()
