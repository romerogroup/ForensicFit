# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:36:25 2021

@author: Pedram Tavadze
used PyChemia Database class as a guide
https://github.com/MaterialsDiscovery/PyChemia/blob/master/pychemia/db/db.py
"""
import os
import numpy as np
import pymongo
import gridfs
from random import choice
from bson.objectid import ObjectId
from ..core import Tape, TapeAnalyzer, Image, Metadata
from ..utils.array_tools import read_bytes_io, write_bytes_io
from collections.abc import Mapping

class ClassMap(Mapping):
    def __init__(self):
        self.mapping = {'material': Tape,
                        'analysis': TapeAnalyzer,
                        'any': Image}
        
    def __contains__(self, x):
        return x in self.mapping

    def __getitem__(self, x):
        if x in self:
            return self.mapping.__getitem__(x)
        else:
            return self.mapping.__getitem__('any')

    def __iter__(self):
        return self.mapping.__iter__()

    def __len__(self):
        return self.mapping.__len__()   
            

class Database:
    def __init__(self,
                 name: str = 'forensicfit',
                 host: str = "localhost",
                 port: int = 27017,
                 username: str = "",
                 password: str = "",
                 verbose: bool = False,
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
        self.class_mapping = ClassMap()
        for x in self.collection_names:
            if x not in self.fs:
                self.add_collection(x)
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

    def add_collection(self, collection: str):
        self.fs[collection] = gridfs.GridFS(self.db, collection)
        return 

    def exists(self, 
               criteria: dict = None, 
               collection: str = None, 
               metadata: Metadata = None) -> ObjectId: #| bool:
        if metadata is not None:
            collection = collection or metadata['mode']
            criteria = {"$and": metadata.to_mongodb_filter()}
        elif criteria is None or collection is None:
            raise Exception("Provide metadata or criteria and collection")
        ret = self.fs[collection].find_one(filter=criteria)
        if ret is not None:
            return ret._id
        else:
            return False

    def insert(self,    
               obj: Image, # | Tape | TapeAnalyzer,
               buffer_type: str = '.npz',
               overwrite: bool = False, 
               skip: bool = False,
               collection : str = None):
        
        collection = collection or obj.metadata['mode']
        criteria = {"$and": obj.metadata.to_mongodb_filter()}
        exists = self.exists(criteria=criteria, 
                             collection=collection,
                             )
        if collection not in self.fs:
            self.add_collection(collection)
        fs = self.fs[collection]
        if overwrite and exists:
            if self.verbose:
                print(f"{obj.metadata.filename} {collection} already exists, overwriting!")
            self.delete(criteria, collection)
        elif skip and exists:
            if self.verbose:
                print(f"{obj.metadata.filename} {collection} already exists, skipping!")
            return exists
        metadata = obj.metadata.to_dict
        metadata['buffer_type'] = buffer_type
        filename = obj.metadata.filename
        _id = fs.put(obj.to_buffer(buffer_type), 
                     filename = filename,
                     metadata = metadata)
        return _id


    def find(self, 
             filter: dict, 
             collection: str = 'analysis', 
             version: int = -1,
             no_cursor_timeout = False,
             ) -> list:
        CLASS = self.class_mapping[collection]
        fs = self.fs[collection]
        queries = fs.find(filter=filter, 
                          no_cursor_timeout=no_cursor_timeout).sort("uploadDate", version)
        ret = []
        if queries.count() != 0:
            for iq in queries:
                ret.append(CLASS.from_buffer(iq.read(), iq.metadata))
        return ret

    # def get(self, 
    #         filename: str = None, 
    #         collection: str = 'analysis', 
    #         version: int = -1, 
    #         **kwargs) -> list:
    #     ret = []
    #     fs = self.fs[collection]
    #     CLASS = CoreClass[collection]
    #     if '_id' in kwargs:
    #         _id = kwargs['_id']
    #         iq = fs.get(ObjectId(_id), version=version)
    #         metadata = iq.metadata
    #         values = read_bytes_io(iq)
    #         ret = [CLASS.from_dict(values, metadata)]
    #     else:
    #         criteria = {'$and':[]}
    #         if filename is not None:
    #             criteria['$and'].append({'filename':filename})
    #         for ic in kwargs:
    #             criteria['$and'].append({ic:kwargs[ic]})
    #         ret = self.find(criteria, collection, version)
    #     return ret

    
    def find_one(self, filter, collection: str = None) -> object:
        """finds one entry that matches the criteria. 
        kwargs must be chosen by filter=

        Parameters
        ----------
        collection: str, optional
            material or analysis, if none is chose it will be selected randomly, by default None

        Returns
        -------
        object
            ForensicFit object e.g. Tape
        """        
        if collection is None :
            collection = choice(list(self.fs))
        CLASS = self.class_mapping[collection]
        fs = self.fs[collection]
        iq = fs.find_one(filter)
        return CLASS.from_buffer(iq.read(), iq.metadata)
        
    def find_with_id(self, _id: str, collection: str) -> object:
        """Retrieves core object based on the MongoDB _id

        Parameters
        ----------
        _id : str
            MongoDB _id
        collection : str
            'material' or 'analysis'

        Returns
        -------
        object
            ForensicFit object e.g. Tape
        """        
        CLASS = self.class_mapping[collection]
        fs = self.fs[collection]
        iq = fs.find_one({'_id':_id if type(_id) is ObjectId else ObjectId(_id)})
        metadata = iq.metadata
        values = read_bytes_io(iq)
        return CLASS.from_dict(values, metadata)
    
    @property
    def count(self):
        ret = {}
        for collection in self.fs:
            ret[collection] = self.fs[collection].find().count()
        return ret

    @property
    def collection_names(self):
        return [x.replace(".files", '')for x in self.db.list_collection_names() if 'files' in x]
        

    def delete(self, criteria: dict, collection: str):
        fs = self.fs[collection]
        queries = fs.find(criteria)
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

