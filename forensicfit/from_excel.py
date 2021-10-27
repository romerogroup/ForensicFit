# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
from p_tqdm import p_map
import pandas as pd
from numpy import array
import tqdm
import multiprocessing
from matplotlib.gridspec import GridSpec
import matplotlib.pylab as plt
from .core import Tape, TapeAnalyzer
from .core import DatasetNumpy
from .database import Database




def chunks(files, nprocessors, args):
    ret = []
    nfiles = len(files)
    nchucks = max(nfiles // nprocessors, 1)
    start = 0
    end = 0
    for i in range(min(nprocessors,nfiles)):
        end = start + nchucks
        ret.append([files[start:end], i, args])
        start = end
    if end != nfiles:
        ret[-1][0].append(files[end:nfiles])
    return ret


def exists(db, filename, side="R", flip_h=False,
           analysis_mode="coordinate_based"):

    return db.gridfs_analysis.exists(
        {
            "$and": [
                {"filename": filename},
                {"metadata.side": side},
                {"metadata.image.flip_h": flip_h},
                {"metadata.analysis_mode": analysis_mode},
            ]
        }
    )


def init_child(lock):
    """
    Provide tqdm with the lock from the parent app.
    This is necessary on Windows to avoid racing conditions.
    """
    tqdm.tqdm.set_lock(lock)




def worker(args):
    df = args[0]
    pos = args[1]
    args = args[2]
    modes = args['modes']
    
    ret = {key: {"X": [], "y": [], 'x_std':[], 'y_std':[], 'quality':[], 'separation_method':[]} for key in modes}
    
    db = Database(args['db_name'], args['host'], args['port'],
                  args['username'], args['password'])

    nfiles = len(df)
    errors = []
    
    # for ientry in tqdm.tqdm(
    #         range(nfiles),
    #         position=pos,
    #         desc="assembling using process %d" %
    #         pos,
    #         leave=True):
    for ientry in range(nfiles):
        _id = df.iloc[ientry]['_id']
        match = ['not_match', 'match'][df.iloc[ientry]['match']]
        query = []
        figs = {}

        for mode in modes:
            fig = plt.figure(constrained_layout=True, figsize=(9,18))
            gs = GridSpec(2, 2, figure=fig, width_ratios=[1]*2, height_ratios=[2]*2)
            figs[mode]={'fig': fig, 'gs': gs}
        for isurface, surface in enumerate(["f", "b"]):
            for itape, tape in enumerate(["1", "2"]):
                name = df.iloc[ientry]["tape_{}{}".format(
                    surface, tape)] + ".tif"
                side = df.iloc[ientry]["side_{}{}".format(surface, tape)]
                if tape == "2":
                    flip_h = bool(df.iloc[ientry]["flip_{}".format(surface)])
                else:
                    flip_h = False
                all_exists = True
                for mode in modes:
                    if not exists(
                        db, name, side=side, flip_h=flip_h, analysis_mode=mode
                    ):
                        all_exists = False
                if all_exists:
                    query.append(
                        db.get_analysis(
                            filename=name,
                            side=side,
                            flip_h=flip_h))
                else:
                    print("Not in the database:", name, side)
                for mode in modes:
                    ax = figs[mode]['fig'].add_subplot(figs[mode]['gs'][isurface, itape])
                    query[-1].plot(mode, ax=ax, reverse_x=not bool(itape))
                    ax.set_title("{}_{}".format(name.split(".")[0],side))
        for mode in modes:
            figs[mode]['fig'].savefig("_{}_{}_{}.png".format(mode, _id, match))
            plt.close(figs[mode]['fig'])
        if len(query) == 4:
            for mode in modes:
                if len(query[0][mode].shape) == 3:
                    for j in range(query[0][mode].shape[0]):
                        temp = [q[mode][j] for q in query]
                        ret[mode]['X'].append(temp)
                        ret[mode]['y'].append(df.iloc[ientry]['match'])
                        ret[mode]['x_std'] = [q.metadata['x_std'] for q in query]
                        ret[mode]['y_std'] = [q.metadata['y_std'] for q in query]
                        ret[mode]['separation_method'] = query[0].metadata['separation_method']
                        ret[mode]['quality'] = query[0].metadata['quality']
                else:
                    temp = [q[mode] for q in query]
                    ret[mode]['X'].append(temp)
                    ret[mode]['y'].append(df.iloc[ientry]['match'])
                    ret[mode]['x_std'].append([q.metadata['x_std'] for q in query])
                    ret[mode]['y_std'].append([q.metadata['y_std'] for q in query])
                    ret[mode]['separation_method'].append(query[0].metadata['separation_method'])
                    ret[mode]['quality'].append(query[0].metadata['quality'])
                    
    for mode in modes:
        extra = {'x_std':ret[mode]['x_std'], 'y_std':ret[mode]['y_std'], 'separation_method':ret[mode]['separation_method'], 'quality':ret[mode]['quality'] }
        ret[mode] = DatasetNumpy(array(ret[mode]['X']), ret[mode]['y'], extra=extra,name=mode)

    return ret


def from_excel(
    excel_file,
    modes=["coordinate_based", "bin_based", "big_picture", "max_contrast"],
    db_name="forensicfit",
    host="localhost",
    port=27017,
    username="",
    password="",
    nprocessors=1,
):

    df = pd.read_excel(excel_file)
    ndata = len(df)
    args = locals()

    if nprocessors == 1:
        worker(chunks(df, 1, args)[0])
    elif nprocessors > 1:
        # multiprocessing.freeze_support()

        # lock = multiprocessing.Lock()
        # if nprocessors > multiprocessing.cpu_count():
        #     nprocessors = multiprocessing.cpu_count()
        # p = multiprocessing.Pool(
        #     nprocessors, initializer=init_child, initargs=(lock,))
        args = chunks(df, nprocessors, args)
        
        rets = p_map(worker, args)
        
        # p.close()
        # p.join()
    
    for mode in modes:
        data = DatasetNumpy(name=mode)
        for ip in rets:
            data += ip[mode]
        data.save(mode)

