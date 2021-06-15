# -*- coding: utf-8 -*-

import os
import tqdm
import multiprocessing
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

    for count in tqdm.tqdm(range(nfiles), position=pos, desc="storing using proccess %d" % pos, leave=True):
        ifile = files[count]
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
        
        for iside in args['side']:
            for iflip in [True, False]:
                if args['ignore_errors']:
                    try :
                        tape = Tape(ifile, label=ifile)
        
                        if args['split']:
                            tape.split_vertical(
                                pixel_index=args['split_position'], pick_side=iside)
                        if iflip:
                            tape.flip_h()
                        
                            
                            analyed_tape = TapeAnalyzer(
                                tape, args['mask_threshold'], args['gaussian_blur'], args['ndivision'], args['auto_crop'], args['calculate_tilt'], False)
                            
                        else:
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
                        if 'weft_based' in args['modes']:
                            analyed_tape.get_weft_based(args['window_background'],
                                                        args['window_tape'],
                                                        args['dynamic_window'],
                                                        args['weft_based_size'],
                                                        args['nsegments'][quality.lower()])
                        if 'big_picture' in args['modes']:
                            analyed_tape.get_weft_based(args['window_tape'],
                                                        args['dynamic_window'],
                                                        args['weft_based_size'],
                                                        nsegments=4)
        
                        if 'max_contrast' in args['modes']:
                            analyed_tape.get_max_contrast(
                                args['window_background'], args['window_tape'], args['max_contrast_size'])
                        db.insert(analyed_tape)
                    except:
                        print("Could not analyze file : %s" % ifile)


def process_directory(
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
        DESCRIPTION. The default is ['coordinate_based','weft_based','big_picture','max_contrast'].
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
    args = locals() #dict(modes=modes,
    #             dynamic_window=dynamic_window,
    #             nsegments=nsegments,
    #             ndivision=ndivision,
    #             window_tape=window_tape,
    #             window_background=window_background,
    #             npoints=npoints,
    #             x_trim_param=x_trim_param,
    #             weft_based_size=weft_based_size,
    #             big_picture_size=big_picture_size,
    #             max_contrast_size=max_contrast_size,
    #             split=split,
    #             side=side,
    #             auto_rotate=auto_rotate,
    #             auto_crop=auto_crop,
    #             gaussian_blur=gaussian_blur,
    #             mask_threshold=mask_threshold,
    #             split_position=split_position,
    #             calculate_tilt=calculate_tilt,
    #             skip=skip,
    #             overwrite=overwrite,
    #             db_name=db_name,
    #             host=host,
    #             port=port,
    #             username=username,
    #             password=password,
    #             ignore_errors=ignore_errors)

    if nprocessors == 1:
        worker(chunks(files, 1, args)[0])
    elif nprocessors > 1:
        multiprocessing.freeze_support()

        lock = multiprocessing.Lock()
        if nprocessors > multiprocessing.cpu_count():
            nprocessors = multiprocessing.cpu_count()
        p = multiprocessing.Pool(
            nprocessors, initializer=init_child, initargs=(lock,))
        args = chunks(files, nprocessors, args)
        p.map(worker, args)

        p.close()
        p.join()
