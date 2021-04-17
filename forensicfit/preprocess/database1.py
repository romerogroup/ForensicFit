# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:36:25 2021

@author: Pedram Tavadze
"""
import os
import h5py
import pandas as pd
import pymongo
from .tape_image import TapeImage


class Database1():
    def __init__(self, 
                 uri="mongodb://localhost:27017/",
                 username='user',
                 password="password",
                 fname='db.hdf5',
                 initiate=False,
                 modes=['coordinate_based', 'weft_based',
                        'big_picture', 'max_contrast'],
                 meta_file=None,
                 src_dir=None,
                 mask_threshold=60,
                 gaussian_blur=(15, 15),
                 dynamic_window=True,
                 nsegments=39,
                 window_tape=100,
                 window_background=50,
                 sizes=dict(coordinate_based=(1000,), weft_based=(300, 30),
                            big_picture=(1200, 30), max_contrast=(1000, 200)),
                 auto_rotate=False,
                 split_position=0.5,
                 verbose=True,
                 plot=False):
        """
        

        Parameters
        ----------
        fname : TYPE, optional
            DESCRIPTION. The default is 'db.hdf5'.
        initiate : TYPE, optional
            DESCRIPTION. The default is False.
        modes : TYPE, optional
            DESCRIPTION. The default is ['coordinate_based', 'weft_based',                        'big_picture', 'max_contrast'].
        meta_file : TYPE, optional
            DESCRIPTION. The default is None.
        src_dir : TYPE, optional
            DESCRIPTION. The default is None.
        mask_threshold : TYPE, optional
            DESCRIPTION. The default is 60.
        gaussian_blur : TYPE, optional
            DESCRIPTION. The default is (15, 15).
        dynamic_window : TYPE, optional
            DESCRIPTION. The default is True.
        nsegments : TYPE, optional
            DESCRIPTION. The default is 39.
        window_tape : TYPE, optional
            DESCRIPTION. The default is 100.
        window_background : TYPE, optional
            DESCRIPTION. The default is 50.
        sizes : TYPE, optional
            DESCRIPTION. The default is dict(coordinate_based=(1000,), weft_based=(300, 30),                            big_picture=(1200, 30), max_contrast=(4800, 30)).
        auto_rotate : TYPE, optional
            DESCRIPTION. The default is False.
        split_position : TYPE, optional
            DESCRIPTION. The default is 0.5.
        verbose : TYPE, optional
            DESCRIPTION. The default is True.
        plot : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        
        self.fname = fname
        self.modes = modes
        self.mask_threshold = mask_threshold
        self.gaussian_blur = gaussian_blur
        self.dynamic_window = dynamic_window
        self.nsegments = nsegments
        self.window_background = window_background
        self.window_tape = window_tape
        self.sizes = sizes
        self.meta_file = meta_file
        self.meta_dataframe = None
        self.src_dir = src_dir
        self.verbose = verbose

        if initiate:
            self.create_db()

    def create_db(self):
        """
        

        Returns
        -------
        None.

        """
        if os.path.exists(self.fname):
            if self.verbose:
                print("%s already exists, changing the name to %s" %
                      (self.fname, self.fname+"_1"))
            self.fname = self.fname+"_1"
        if self.verbose:
            print("Openning the metadata file.")
        self.meta_dataframe = pd.read_excel(self.meta_file)
        ndata = 4  # self.meta_dataframe.shape[0]
        if self.verbose:
            print("creating the database file %s" % self.fname)
        wf = h5py.File(self.fname, 'w')
        for mode in self.modes:
            if self.verbose:
                print("creating dataset %s" % mode)
            size = self.sizes[mode]
            size = tuple([ndata, 4]+list(size))
            wf.create_dataset(mode, shape=size, dtype=float)

        wf.create_dataset('match', shape=(ndata,), dtype=bool)
        
        for idata in range(ndata):
            df = self.meta_dataframe.iloc[idata]
            tf1 = TapeImage(fname=self.src_dir+os.sep+df['tape_f1']+'.tif', split_side=df['side_f1'],
                            mask_threshold=self.mask_threshold, gaussian_blur=self.gaussian_blur)
            tf1.auto_crop_y()
            wf['max_contrast'][idata, 0, :, :] = tf1.max_contrast(window_background=self.window_background,
                                                      window_tape=self.window_tape, size=self.sizes["max_contrast"])[:, :]
            tf2 = TapeImage(fname=self.src_dir+os.sep+df['tape_f2']+'.tif', flip=df["flip_f"], split_side=df['side_f2'],
                            mask_threshold=self.mask_threshold, gaussian_blur=self.gaussian_blur)
            tf2.auto_crop_y()
            wf['max_contrast'][idata, 1, :, :] = tf2.max_contrast(window_background=self.window_background,
                                                                  window_tape=self.window_tape, size=self.sizes["max_contrast"])[:, :]
            tb1 = TapeImage(fname=self.src_dir+os.sep+df['tape_b1']+'.tif', split_side=df['side_b1'],
                            mask_threshold=self.mask_threshold, gaussian_blur=self.gaussian_blur)
            tb1.auto_crop_y()
            wf['max_contrast'][idata, 2, :, :] = tb1.max_contrast(window_background=self.window_background,
                                                                  window_tape=self.window_tape, size=self.sizes["max_contrast"])[:, :]
            tb2 = TapeImage(fname=self.src_dir+os.sep+df['tape_b2']+'.tif', flip=df["flip_b"], split_side=df['side_b2'],
                            mask_threshold=self.mask_threshold, gaussian_blur=self.gaussian_blur)
            tb2.auto_crop_y()
            wf['max_contrast'][idata, 3, :, :] = tb2.max_contrast(window_background=self.window_background,
                                                                  window_tape=self.window_tape, size=self.sizes["max_contrast"])[:, :]
            
        wf.close()
