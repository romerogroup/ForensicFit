# -*- coding: utf-8 -*-

import os
import matplotlib.pylab as plt
import numpy as np
import tqdm
import multiprocessing
from matplotlib.gridspec import GridSpec
from .core import Tape, TapeAnalyzer
from .database import Database


def chunks(files, nprocessors, args):
    ret = []
    nfiles = len(files)
    nchucks = nfiles//nprocessors
    start = 0
    end = 0
    for i in range(nprocessors):
        end = start + nchucks
        ret.append([files[start:end], i, args])
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
    pos = args[1]
    args = args[2]

    db = Database(args['db_name'], args['host'], args['port'],
                  args['username'], args['password'])

    nfiles = len(files)
    errors =[]
    for count in tqdm.tqdm(range(nfiles), position=pos, desc="storing using proccess %d" % pos, leave=True):
        ifile = files[count]
        if len(ifile.split('.')) == 1:

            continue
        if ifile.split('.')[1] not in ['tif', 'jpg', 'bmp', 'png']:

            continue
        if db.exists_analysis(ifile) and args['skip']:
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

        fig = plt.figure(constrained_layout=True, figsize=(21,9))
        gs = GridSpec(3, 16, figure=fig, width_ratios=[1]*16, height_ratios=[3]*3)

        for iside, side in enumerate(args['side']):
            for iflip, flip in enumerate([True, False]):

                tape = Tape(ifile, label=ifile)
                if side == 'R' and flip:
                    ax = fig.add_subplot(gs[0, :])
                    tape.plot(ax=ax)


                if args['split']:
                    tape.split_vertical(
                        pixel_index=args['split_position'], pick_side=side)

                if flip:
                    tape.flip_h()
                pos = ([2,0][iside]+[1, 0][iflip])*4
                ax = fig.add_subplot(gs[1, pos:pos+4])
                tape.plot(ax=ax)
                try:
                    analyed_tape = TapeAnalyzer(
                            tape, args['mask_threshold'], args['gaussian_blur'], args['ndivision'], args['auto_crop'], args['calculate_tilt'], False)
                    analyed_tape.add_metadata("quality", quality)
                    analyed_tape.add_metadata(
                            "separation_method", separation_method)
                    analyed_tape.add_metadata("streched", streched)
                    # analyed_tape.add_metadata("side", side)

                    if 'coordinate_based' in args['modes']:
                        analyed_tape.get_coordinate_based(
                            args['npoints'], args['x_trim_param'])
                        ax = fig.add_subplot(gs[2,pos])
                        analyed_tape.plot("coordinate_based", ax=ax)
                        # ax.set_title("coordinate_based")
                    if 'bin_based' in args['modes']:
                        analyed_tape.get_bin_based(args['window_background'],
                                                   args['window_tape'],
                                                   args['dynamic_window'],
                                                   args['bin_based_size'],
                                                   args['nsegments'][quality.lower()])
                        ax = fig.add_subplot(gs[2,pos+1])
                        analyed_tape.plot("bin_based", ax=ax)
                        # ax.set_title("bin_based")
                    if 'big_picture' in args['modes']:
                        analyed_tape.get_bin_based(args['window_tape'],
                                                   args['dynamic_window'],
                                                   args['big_picture'],
                                                   nsegments=4)
                        ax = fig.add_subplot(gs[2,pos+2])
                        analyed_tape.plot('big_picture', ax=ax)
                        # ax.set_title('big_picture')
                    if 'max_contrast' in args['modes']:
                        analyed_tape.get_max_contrast(
                            args['window_background'], 800, size=args['max_contrast_size'])
                        ax = fig.add_subplot(gs[2,pos+3])
                        analyed_tape.plot('max_contrast', ax=ax)
                        # ax.set_title('max_contrast')
                    db.insert(analyed_tape)
                    control_name = ifile.split('.')[0]+'.png'
                except:
                    errors.append(ifile.split('.')[0]+'-'+side+'-'+str(flip))
                    control_name = "__error_"+ifile.split('.')[0]+'.png'
        plt.savefig(control_name)
        plt.close()
    return errors



def process_directory(
        dir_path='.',
        modes=['coordinate_based', 'bin_based',
               'big_picture', 'max_contrast'],
        dynamic_window=True,
        nsegments={"h": 56, "m": 36, "l": 32},
        ndivision=6,
        window_tape=100,
        window_background=50,
        npoints=1000,
        x_trim_param=3,
        bin_based_size=(512, 256),
        big_picture_size=(1024, 256),
        max_contrast_size=(2048, 512),
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
        ignore_errors=False,
        nprocessors=1):
    """


    Parameters
    ----------
    dir_path : TYPE, optional
        DESCRIPTION. The default is '.'.
    output_format : TYPE, optional
        DESCRIPTION. The default is 'json'.
    modes : TYPE, optional
        DESCRIPTION. The default is ['coordinate_based','bin_based','big_picture','max_contrast'].
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

    if split:
        if side == 'both':
            side = ['L', 'R']
        else:
            side = [side]
    else:
        side = ['L']
    files = os.listdir(dir_path)
    args = locals() 
    cwd = os.getcwd()
    os.chdir(dir_path)
    if nprocessors == 1:
        worker(chunks(files, 1, args)[0])
    elif nprocessors > 1:
        multiprocessing.freeze_support()

        lock = multiprocessing.Lock()
        # if nprocessors > multiprocessing.cpu_count():
        #     nprocessors = multiprocessing.cpu_count()
        p = multiprocessing.Pool(
            nprocessors, initializer=init_child, initargs=(lock,))
        args = chunks(files, nprocessors, args)
        errors = p.map(worker, args)

        p.close()
        p.join()
    os.chdir(cwd)
    wf = open("errors.log", 'w')
    for err in errors:
        for i in np.unique(err):
            wf.write(i+os.linesep)
    wf.close
