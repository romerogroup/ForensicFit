# -*- coding: utf-8 -*-

import os
import pandas as pd
from numpy import array
import multiprocessing
from matplotlib.gridspec import GridSpec
import matplotlib.pylab as plt
from bson.objectid import ObjectId
from .core import Tape, TapeAnalyzer
from .core import DatasetNumpy
from .database import Database


def chunks(files, nprocessors, args):
    ret = []
    nfiles = len(files)
    print("Total queries to be made: {}".format(nfiles))
    print("Assigning jobs to cpus")
    nchucks = max(nfiles // nprocessors, 1)
    start = 0
    end = 0
    for i in range(min(nprocessors, nfiles)):
        end = start + nchucks
        ret.append([files[start:end], i, args])
        print(
            "cpu #{: >4} - from {: >4} to {: >4} - total {: >4}".format(i, start, end, nchucks))
        start = end
    if end != nfiles:
        ret[-1][0].append(files[end:nfiles])
        print("cpu #{: >4} - from {: >4} to {: >4} - total {: >4}".format(i +
              1, end, nfiles, nfiles - end))
    return ret



def exists(db, filename, side="R", flip_h=False,
           analysis=["coordinate_based"]):
    and_list = [
                {"filename": filename},
                {"metadata.side": side},
                {"metadata.image.flip_h": flip_h}]
    for imode in analysis:
        and_list.append({"metadata.analysis.{}".format(imode):{"$exists":True}})
    return db.exists(criteria = {
            "$and": and_list
    }, mode='analysis',
    )

# def init_child(lock):
#     """
#     Provide tqdm with the lock from the parent app.
#     This is necessary on Windows to avoid racing conditions.
#     """
#     tqdm.tqdm.set_lock(lock)


def worker(args):
    df = args[0]
    pos = args[1]
    args = args[2]
    modes = args['modes']

    ret = {
        key: {
            "X": [],
            "y": [],
            'x_std': [],
            'y_std': [],
            'quality': [],
            'separation_method': []} for key in modes}

    db = Database(args['db_name'], args['host'], args['port'],
                  args['username'], args['password'], verbose=False)

    nfiles = len(df)
    errors = []

    for ientry in range(nfiles):
        _id_excel = df.iloc[ientry]['_id']
        match = ['not_match', 'match'][df.iloc[ientry]['match']]
        query = []
        figs = {}
        for mode in modes:
            fig = plt.figure(constrained_layout=True, figsize=(9, 18))
            gs = GridSpec(2, 2, figure=fig, width_ratios=[
                          1] * 2, height_ratios=[2] * 2)
            figs[mode] = {'fig': fig, 'gs': gs}
        all_exists = []
        for isurface, surface in enumerate(["f", "b"]):
            for itape, tape in enumerate(["1", "2"]):
                name = df.iloc[ientry]["tape_{}{}".format(
                    surface, tape)] + ".tif"
                side = df.iloc[ientry]["side_{}{}".format(surface, tape)]
                if tape == "2":
                    flip_h = bool(df.iloc[ientry]["flip_{}".format(surface)])
                else:
                    flip_h = False
                ex = exists(
                        db, name, side=side, flip_h=flip_h, analysis=modes
                    )
                
                all_exists.append(ex)
                if not ex:
                    print("Not in the database:", name, side)

        

        if not all(all_exists):
            for mode in modes:
                figs[mode]['fig'].savefig("PNG{os.sep}_error_{mode}_{_id_excel}_{match}.png")
                plt.close(figs[mode]['fig'])
            continue
        

        for tape_id in all_exists:
            tape_analysis = db.find_one(mode='analysis', filter={"_id":tape_id})
            query.append(tape_analysis)
            
            for mode in modes:
                ax = figs[mode]['fig'].add_subplot(
                        figs[mode]['gs'][isurface, itape])
                tape_analysis.plot(mode, ax=ax, reverse_x=not bool(itape))
                ax.set_title("{}_{}".format(name.split(".")[0], side))

        for mode in modes:
            figs[mode]['fig'].savefig(f"PNG{os.sep}{mode}_{_id_excel}_{match}.png")
            plt.close(figs[mode]['fig'])
    
        for mode in modes:
            if mode in ['bin_based', 'big_picture']:
                for j in range(query[0][mode].shape[0]):
                    temp = [q[mode][j] for q in query]
                    ret[mode]['X'].append(temp)
                    ret[mode]['y'].append(df.iloc[ientry]['match'])
                    ret[mode]['x_std'] = [q.metadata['x_std']
                                          for q in query]
                    ret[mode]['y_std'] = [q.metadata['y_std']
                                          for q in query]
                    ret[mode]['separation_method'] = query[0].metadata['separation_method']
                    ret[mode]['quality'] = query[0].metadata['quality']
            elif mode == 'max_contrast':
                temp = [q[mode] for q in query]
                ret[mode]['X'].append(temp)
                ret[mode]['y'].append(df.iloc[ientry]['match'])
                ret[mode]['x_std'].append(
                    [q.metadata['x_std'] for q in query])
                ret[mode]['y_std'].append(
                    [q.metadata['y_std'] for q in query])
                ret[mode]['separation_method'].append(
                    query[0].metadata['separation_method'])
                ret[mode]['quality'].append(query[0].metadata['quality'])
            elif mode == 'coordinate_based':
                temp = [q[mode] for q in query]                
                ret[mode]['X'].append(temp)
                ret[mode]['y'].append(df.iloc[ientry]['match'])
                ret[mode]['x_std'].append(
                    [q.metadata['x_std'] for q in query])
                ret[mode]['y_std'].append(
                    [q.metadata['y_std'] for q in query])
                ret[mode]['separation_method'].append(
                    query[0].metadata['separation_method'])
                ret[mode]['quality'].append(query[0].metadata['quality'])                
                
    for mode in modes:
        extra = {
            'x_std': array(ret[mode]['x_std']),
            'y_std': array(ret[mode]['y_std']),
            'separation_method': array(ret[mode]['separation_method']),
            'quality': array(ret[mode]['quality'])}
        ret[mode] = DatasetNumpy(
            array(
                ret[mode]['X']), array(
                ret[mode]['y']), extra=extra, name=mode)

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
    if not os.path.exists("PNG"):
        os.mkdir("PNG")
    if nprocessors == 1:
        rets = [worker(chunks(df, 1, args)[0])]
    elif nprocessors > 1:
        # multiprocessing.freeze_support()

        # lock = multiprocessing.Lock()
        if nprocessors > multiprocessing.cpu_count():
            nprocessors = multiprocessing.cpu_count()
        p = multiprocessing.Pool(nprocessors)
            # nprocessors, initializer=init_child, initargs=(lock,))

        args = chunks(df, nprocessors, args)

        rets = p.map(worker, args)

        p.close()
        p.join()
    for mode in modes:
        data = rets[0][mode]
        for ip in range(1, len(rets)):
            data += rets[ip][mode]
        data.save(mode)
