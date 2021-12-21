# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:36:25 2021

@author: Pedram Tavadze
used PyChemia Database class as a guide
https://github.com/MaterialsDiscovery/PyChemia/blob/master/pychemia/db/db.py
"""
import os
# import h5py
import pandas as pd
import numpy as np
import pymongo
import gridfs
from random import choice
from bson.objectid import ObjectId
from ..core import Tape, TapeAnalyzer
from ..utils.array_tools import read_bytes_io, write_bytes_io

CoreClass = {'material': Tape,
             'analysis': TapeAnalyzer}


class Database:
    def __init__(self,
                 name: str = 'forensicfit',
                 host: str = "localhost",
                 port: int = 27017,
                 username: str = "",
                 password: str = "",
                 verbose: bool = True,
                 **kwargs):

        self.name = name
        self.host = host
        self.port = port

        self.username = username
        self.password = password
        self.verbose = verbose

        if len(password) != 0:
            self.password = ":"+password+"@"
        else:
            if len(username) != 0:
                self.username += "@"
        self.uri = "mongodb://%s%s%s:%d" % (username, password, host, port)
        self.client = pymongo.MongoClient(self.uri)
        self.db = self.client[name]
        self.entries = self.db.forensicfit_entries
        self.fs = {}
        self.fs['material'] = gridfs.GridFS(self.db, "material")
        self.fs['analysis'] = gridfs.GridFS(self.db, "analysis")
        self.db_info = {"Database Name": self.name,
                        "Host": self.host,
                        "Port": self.port,
                        "User": self.username}
        if self.verbose:
            print("----------------")
            print("connected to:")
            print(self)


    def disconnect(self):
        """Closes the connection with the mongodb Client.

        """
        
        self.client.close()

    def __str__(self):
        ret = "MongoDB\n"
        ret += "----------------\n"
        for key in self.db_info:
            ret += "{:<15}: {}\n".format(key, self.db_info[key])
        return ret


    def exists(self, criteria: dict, mode: str) -> bool:
        ret = self.fs[mode].find_one(filter=criteria)
        if ret is not None:
            return ret._id
        else:
            return False

    def insert(self, obj: object, overwrite: bool = False, skip: bool = False, save_minimal: bool = True):
        mode = obj.metadata['mode']
        criteria = {"$and":dict2mongo_query(obj.metadata, 'metadata')}
        exists = self.exists(mode=mode,
                             criteria=criteria)
        fs = self.fs[mode]
        if skip and exists:
            if self.verbose:
                print("{} {} already exists, skipping!".format(obj.filename, mode))
            return exists._id
        if overwrite and exists:
            if self.verbose:
                print("{} {} already exists, overwriting!".format(obj.filename, mode))
            self.delete(criteria, mode)
        output = write_bytes_io(obj.values)
        metadata = obj.metadata
        filename = obj.filename.split("/")[-1].split("\\")[-1]
        _id = fs.put(output, filename=filename, metadata=metadata)
        return _id


    def find(self, criteria: dict, mode: str = 'analysis', version: int = -1) -> list:
        CLASS = CoreClass[mode]
        queries = self.fs[mode].find(criteria).sort("uploadDate", version)
        ret = []
        if queries.count() != 0:
            for iq in queries:
                metadata = iq.metadata
                values = read_bytes_io(iq)
                ret.append(CLASS.from_dict(values, metadata))
        return ret


    def get(self, filename: str = None, mode: str = 'analysis', version: int = -1, **kwargs) -> list:
        ret = []
        fs = self.fs[mode]
        CLASS = CoreClass[mode]
        if '_id' in kwargs:
            _id = kwargs['_id']
            iq = fs.get(ObjectId(_id), version=version)
            metadata = iq.metadata
            values = read_bytes_io(iq)
            ret = [CLASS.from_dict(values, metadata)]
        else:
            criteria = {'$and':[]}
            if filename is not None:
                criteria['$and'].append({'filename':filename})
            for ic in kwargs:
                criteria['$and'].append({ic:kwargs[ic]})
            ret = self.find(criteria, mode, version)
        return ret

    
    def find_one(self, mode: str = None, **kwargs) -> object:
        if mode is None :
            mode = choice(list(self.fs))
        CLASS = CoreClass[mode]        
        fs = self.fs[mode]
        iq = fs.find_one(**kwargs)
        metadata = iq.metadata
        values = read_bytes_io(iq)
        return CLASS.from_dict(values, metadata)
      
    
    @property
    def count(self):
        ret = {}
        for mode in self.fs:
            ret[mode] = self.fs[mode].find().count()
        return ret

    @property
    def collection_names(self):
        return [x.replace(".files", '')for x in self.db.collection_names() if 'files' in x]
        

    def delete(self, criteria: dict, mode: str):
        fs = self.fs[mode]
        queries = self.fs[mode].find(criteria)
        for iq in queries:
            fs.delete(iq._id)
        return 
        
    def delete_database(self):
        self.client.drop_database(self.name)

    @property
    def connected(self):
        try:
            self.client.server_info()  # force connection on a request as the
            return True
        except pymongo.errors.ServerSelectionTimeoutError as err:
            print(err)
            return False
        
    @property
    def server_info(self):
        return self.client.server_info()

def dict2mongo_query(inp: dict, previous_key: str = '') -> dict:
    ret = []
    for key in inp:
        if  isinstance(inp[key], dict) :
            if len(inp[key]) != 0 :
                ret.append(dict2mongo_query(inp[key],
                                            previous_key=previous_key+'.'+key,
                                            ))
            else:
                ret.append({previous_key+'.'+key:inp[key]})
        else:
            ret.append({previous_key+'.'+key:inp[key]})
    ret_p = []
    for item_1 in ret:
        if isinstance(item_1, list):
            for item_2 in item_1:
                ret_p.append(item_2)
        else:
            ret_p.append(item_1)
    return ret_p
            

def list_databases(host='localhost',
                   port=27017,
                   ):
    uri = "mongodb://%s%s%s:%d" % (username, password, host, port)
    client = pymongo.MongoClient(uri)
    database_names = client.list_database_names()
    return database_names

