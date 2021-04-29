# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:36:25 2021

@author: Pedram Tavadze
"""
import os
import h5py
import pandas as pd
import pymongo
# from ..preprocess import TapeImage


class Database:
    def __init__(self,
                 db_name='forensicfit',
                 host='localhost',
                 port=27017,
                 username="",
                 password=""):

        self.db_name = db_name
        self.host = host
        self.port = port

        self.username = username
        self.password = password
        if len(password) != 0:
            self.password = ":"+password+"@"
        else:
            if len(username) != 0:
                self.username += "@"
        self.uri = "mongodb://%s%s%s:%d" % (username, password, host, port)
        self.client = pymongo.MongoClient(self.uri)
        self.db = self.client[db_name]
        self.collection = self.db["image"]
        self.db_info = {"DB Name": self.db_name,
                        "Host": self.host,
                        "Port": self.port,
                        "User": self.username}

    def __str__(self):
        ret = ""
        for key in self.db_info:
            ret += "{:<7}    :  {}\n".format(key, self.db_info[key])
        return ret

    def insert(self, item):
        self.collection.insert_one(item)
            
            
            

    def create_db(self,
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
