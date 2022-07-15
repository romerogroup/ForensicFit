# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:36:25 2021

@author: Pedram Tavadze
used PyChemia Database class as a guide
https://github.com/MaterialsDiscovery/PyChemia/blob/master/pychemia/db/db.py
"""
import subprocess
import numpy as np
import pymongo
import bson
import gridfs
import pathlib
from pathlib import Path
from random import choice
from bson.objectid import ObjectId
from collections.abc import Mapping
from typing import Callable, List, Optional, Union
from ..core import Tape, TapeAnalyzer, Image, Metadata
from ..utils.array_tools import read_bytes_io, write_bytes_io


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
               filter: dict = None, 
               collection: str = None, 
               metadata: Metadata = None) -> ObjectId: #| bool:
        if collection not in self.fs:
            return False
        if metadata is not None:
            collection = collection or metadata['mode']
            filter = {"$and": metadata.to_mongodb_filter()}
        elif filter is None or collection is None:
            raise Exception("Provide metadata or filter and collection")
        ret = self.fs[collection].find_one(filter=filter)
        if ret is not None:
            return ret._id
        else:
            return False

    def insert(self,    
               obj: Image or Tape or TapeAnalyzer,
               ext: str = '.png',
               overwrite: bool = False, 
               skip: bool = False,
               collection : str = None):
        
        collection = collection or obj.metadata['mode']
        filter = {"$and": obj.metadata.to_mongodb_filter()}
        exists = self.exists(filter=filter, 
                             collection=collection,
                             )
        if collection not in self.fs:
            self.add_collection(collection)
        fs = self.fs[collection]
        if overwrite and exists:
            if self.verbose:
                print(f"{obj.metadata.filename} {collection} already exists, overwriting!")
            self.delete(filter, collection)
        elif skip and exists:
            if self.verbose:
                print(f"{obj.metadata.filename} {collection} already exists, skipping!")
            return exists
        metadata = obj.metadata.to_serial_dict
        metadata['ext'] = ext
        filename = obj.metadata.filename
        _id = fs.put(obj.to_buffer(ext), 
                     filename = filename,
                     metadata = metadata)
        return _id


    def find(self,
             filter: dict, 
             collection: str = 'analysis',
             ext:str = '.npz',
             version: int = -1,
             no_cursor_timeout: bool = False,
             ) -> list:
        Class = self.class_mapping[collection]
        fs = self.fs[collection]

        ret = []
        if self.count_documents(filter, collection) != 0 :
            queries = fs.find(filter=filter, 
                            no_cursor_timeout=no_cursor_timeout).sort("uploadDate", version)
            for iq in queries:
                ret.append(Class.from_buffer(iq.read(), iq.metadata))
        return ret


    def map_to(self,
               func: Callable,
               filter: dict, 
               collection_source: str,
               collection_target,
               verbose: bool=True,
               no_cursor_timeout: bool=False):
        Class = self.class_mapping[collection_source]
        fs = self.fs[collection_source]
        
        queries = fs.find(filter=filter, 
                        no_cursor_timeout=no_cursor_timeout)
        for iq in queries:
            obj = Class.from_buffer(iq.read(), iq.metadata)
            ext = obj.metadata['ext']
            if verbose:
                print(iq.filename)
            self.insert(func(obj), ext=ext, collection=collection_target)
            
                
    
    
    def find_one(self, filter = None, collection: str = None) -> object:
        """finds one entry that matches the filter. 
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
        Class = self.class_mapping[collection]
        fs = self.fs[collection]
        iq = fs.find_one(filter)
        if iq is not None:
            return Class.from_buffer(iq.read(), iq.metadata)
        else:
            raise ValueError(f'No entry found with the filter: {str(filter)}')
        
        
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

        Class = self.class_mapping[collection]
        fs = self.fs[collection]
        iq = fs.find_one({'_id':_id if type(_id) is ObjectId else ObjectId(_id)})
        return Class.from_buffer(iq.read(), iq.metadata)
    
    def filter_with_metadata(self, 
                             inp: List[str], 
                             filter: dict, 
                             collection: str):
        fs = self.fs[collection]
        query = fs.find({'$and':
                        [
                            filter, 
                            {'filename':{'$in':inp}}
                            ]
            })
        db_fnames = [iq.metadata['filename'] for iq in query]
        ret = []
        for i, filename in enumerate(inp):
            if filename in db_fnames:
                ret.append(i)
        return ret
    
    def count_documents(self, filter: dict, collection: str):
        if collection not in self.fs:
            return 0
        fs = self.fs[collection]
        cursor = fs.find()
        return cursor.collection.count_documents(filter=filter)
        
    def export_to_files(self,
                        destination: str,
                        filter: dict,
                        collection: str,
                        ext: str='.png',
                        verbose: bool=True,
                        no_cursor_timeout: bool=False):
        Class = self.class_mapping[collection]
        fs = self.fs[collection]
        queries = fs.find(filter=filter, 
                        no_cursor_timeout=no_cursor_timeout)
        dst = Path(destination)
        dst.mkdir(exist_ok=True)
        for iq in queries:
            obj = Class.from_buffer(iq.read(), iq.metadata)
            ext = obj.metadata['ext']
            path = dst / iq.filename
            if verbose:
                print(path.as_posix())
            obj.to_file(path.with_suffix(ext))




    def drop_collection(self, collection: str):
        for x in ['files', 'chunks']:
            print(f'drop {collection}.{x}')
            self.db.drop_collection(f'{collection}.{x}')
        return

    def delete(self, filter: dict, collection: str):
        fs = self.fs[collection]
        queries = fs.find(filter)
        for iq in queries:
            fs.delete(iq._id)
        return 
        
    def delete_database(self):
        self.client.drop_database(self.name)

    @property
    def collection_names(self):
        return [x.replace(".files", '')for x in self.db.list_collection_names() if 'files' in x]

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
                   username='',
                   password='',
                   ):
    uri = "mongodb://%s%s%s:%d" % (username, password, host, port)
    client = pymongo.MongoClient(uri)
    database_names = client.list_database_names()
    return database_names


def dump(db: Optional[str] = None, 
         host: Optional[str] = None,
         port: Optional[int] = None,
         username: Optional[str] = None,
         password: Optional[str] = None,
         out: Optional[str] = None,
         collection: Optional[str] = None,
         ):
    command = 'mongodump '
    tags = locals()
    for itag in tags:
        if tags[itag] is not None and itag != 'command':
            command += f'--{itag}={tags[itag]} '
    print(command)
    print(subprocess.check_output(command, shell=True))
    
def restore(path: Optional[str] = None,
            db: Optional[str] = None, 
            host: Optional[str] = None,
            port: Optional[int] = None,
            username: Optional[str] = None,
            password: Optional[str] = None,
            collection: Optional[str] = None,
         ):
    command = 'mongorestore '
    tags = locals()
    for itag in tags:
        if tags[itag] is not None and itag not in ['command', 'path']:
            command += f'--{itag}={tags[itag]} '
    command += path
    print(command)
    subprocess.check_output(command, shell=True)