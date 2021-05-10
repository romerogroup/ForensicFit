# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:36:25 2021

@author: Pedram Tavadze
used PyChemia Database class as a guide  
https://github.com/MaterialsDiscovery/PyChemia/blob/master/pychemia/db/db.py
"""
import os
import h5py
import pandas as pd
import numpy as np
import pymongo
import gridfs
import io
from bson.objectid import ObjectId
from ..core import Tape, TapeAnalyzer
# from ..preprocess import TapeImage


class Database:
    def __init__(self,
                 name='forensicfit',
                 host='localhost',
                 port=27017,
                 username="",
                 password=""):

        self.name = name
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
        self.db = self.client[name]
        self.entries = self.db.forensicfit_entries

        self.gridfs_item = gridfs.GridFS(self.db, "items")
        self.gridfs_analysis = gridfs.GridFS(self.db, "analysis")
        self.db_info = {"Database Name": self.name,
                        "Host": self.host,
                        "Port": self.port,
                        "User": self.username}

    def __str__(self):
        ret = ""
        for key in self.db_info:
            ret += "{:<7}    :  {}\n".format(key, self.db_info[key])
        return ret

    def insert_item(self, item):
        if item.metadata['mode'] == 'analysis':
            for key in item.values:
                if type(item[key]) is np.ndarray:
                    output = io.BytesIO()
                    np.save(output, item.values[key])
                    # # This is to erase the other types of analysis
                    metadata = item.metadata.copy()
                    # metadata['analysis'] = {}
                    metadata['analysis_mode'] = key
                    # if key != 'image':

                    #     if key in item.metadata['analysis']:
                    #         metadata['analysis'][key] = item.metadata['analysis'][key]
                    # metadata = item.metadata
                    self.gridfs_analysis.put(output.getvalue(), filename=item.label,
                                             metadata=metadata)
        elif item.metadata['mode'] == 'item':
            for key in item.values:
                if type(item[key]) is np.ndarray:
                    output = io.BytesIO()
                    np.save(output, item.values[key])
                    self.gridfs_item.put(output.getvalue(), filename=item.filename,
                                         metadata=item.metadata)

    def query(self, criteria={}, version=-1):
        ret = None
        queries = self.gridfs_analysis.find(
            criteria).sort("uploadDate", version)
        if queries.count() != 0:
            metadata = {}
            values = {}
            for iq in queries:
                # if len(metadata) == 0:
                #     metadata = iq.metadata
                # if len(iq.metadata["analysis"]) != 0:
                #     metadata['analysis'][iq.metadata['analysis_mode']
                #                          ] = iq.metadata['analysis'][iq.metadata['analysis_mode']]
                values[iq.metadata['analysis_mode']] = np.load(
                    io.BytesIO(iq.read()))
                # values['metadata'] = metadata
            values['metadata'] = iq.metadata
            ret = TapeAnalyzer.from_dict(values)
        return ret

    def get_item(self, filename=None, _id=None, mode='analysis', side="R", version=-1):
        ret = None
        if mode == 'analysis':
            if _id is not None:
                iq = self.gridfs_analysis.get(ObjectId(_id), version=version)
                ret = {'data': np.load(io.BytesIO(iq.read())),
                       'metadata': iq.metadata}
            else:
                queries = self.gridfs_analysis.find(
                    {"filename": filename, "metadata.split_vertical.side": side}).sort("uploadDate", version)
                if queries.count() != 0:
                    metadata = {}
                    values = {}
                    for iq in queries:
                        # if len(metadata) == 0:
                        #     metadata = iq.metadata
                        # if len(iq.metadata["analysis"]) != 0:
                        #     metadata['analysis'][iq.metadata['analysis_mode']
                        #                          ] = iq.metadata['analysis'][iq.metadata['analysis_mode']]
                        values[iq.metadata['analysis_mode']] = np.load(
                            io.BytesIO(iq.read()))
                        # values['metadata'] = metadata
                    values['metadata'] = iq.metadata
                    ret = TapeAnalyzer.from_dict(values)
                    # ret = values
        elif mode == 'item':
            if _id is not None:
                iq = self.gridfs_analysis.get(ObjectId(_id))
                ret = {'data': np.load(io.BytesIO(iq.read())),
                       'metadata': iq.metadata}
            else:
                queries = self.gridfs_item.find(
                    {"filename": filename, "metadata.split_vertical.side": side}).sort("uploadDate", version).limit(1)

                iq = next(queries, None)
                if iq is not None:

                    metadata = {}
                    values = {}
                    values['image'] = np.load(
                        io.BytesIO(iq.read()))
                    values['filename'] = filename
                    values['label'] = iq.metadata['label']
                    values['metadata'] = iq.metadata

                    ret = Tape.from_dict(values=values)

        return ret

    def delete_database(self):
        self.client.drop_database(self.name)

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
            wf.create_dataset(mode, shape=size, dmode=float)

        wf.create_dataset('match', shape=(ndata,), dmode=bool)

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
