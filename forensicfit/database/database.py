# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:36:25 2021

@author: Pedram Tavadze
used PyChemia Database class as a guide
https://github.com/MaterialsDiscovery/PyChemia/blob/master/pychemia/db/db.py
"""
import subprocess
import pymongo
import gridfs
from pathlib import Path
from random import choice
from bson.objectid import ObjectId
from collections.abc import Mapping
from typing import Callable, List, Optional, Union
from ..core import Tape, TapeAnalyzer, Image, Metadata
from ..utils.array_tools import read_bytes_io, write_bytes_io


class ClassMap(Mapping):
    """
    A custom mapping class that maps string keys to specific classes.

    This class is a subclass of `collections.abc.Mapping` and provides a custom mapping
    between string keys and classes. The keys 'material', 'analysis', and 'any' are mapped
    to the classes `Tape`, `TapeAnalyzer`, and `Image` respectively.

    Attributes
    ----------
    mapping : dict
        The internal dictionary that stores the mapping between keys and classes.

    Methods
    -------
    __contains__(x: str) -> bool
        Check if `x` is a key in the mapping.
    __getitem__(x: str) -> Union[Tape, TapeAnalyzer, Image]
        Get the class associated with the key `x`. If `x` is not a key in the mapping, return the class associated with the key 'any'.
    __iter__() -> Iterator
        Return an iterator over the keys in the mapping.
    __len__() -> int
        Return the number of key-value pairs in the mapping.
    """

    def __init__(self):
        """
        Initialize a new instance of ClassMap.
        """
        self.mapping = {'material': Tape,
                        'analysis': TapeAnalyzer,
                        'any': Image}

    def __contains__(self, x):
        """
        Check if `x` is a key in the mapping.

        Parameters
        ----------
        x : str
            The key to check.

        Returns
        -------
        bool
            True if `x` is a key in the mapping, False otherwise.
        """
        return x in self.mapping

    def __getitem__(self, x):
        """
        Get the class associated with the key `x`.

        If `x` is not a key in the mapping, return the class associated with the key 'any'.

        Parameters
        ----------
        x : str
            The key to get the associated class for.

        Returns
        -------
        Union[Tape, TapeAnalyzer, Image]
            The class associated with the key `x`, or the class associated with the key 'any' if `x` is not a key in the mapping.
        """
        if x in self:
            return self.mapping.__getitem__(x)
        else:
            return self.mapping.__getitem__('any')

    def __iter__(self):
        """
        Return an iterator over the keys in the mapping.

        Returns
        -------
        Iterator
            An iterator over the keys in the mapping.
        """
        return self.mapping.__iter__()

    def __len__(self):
        """
        Return the number of key-value pairs in the mapping.

        Returns
        -------
        int
            The number of key-value pairs in the mapping.
        """
        return self.mapping.__len__()


class Database:
    """
    .. _Database:

    Database
    ========

    This class provides an interface to interact with the MongoDB database.

    The Database class encapsulates the MongoDB client, and provides methods 
    to query and manipulate the data stored in the MongoDB collections.

    Parameters
    ----------
    name : str, optional
        The name of the database, defaults to 'forensicfit'.
    host : str, optional
        The host IP address or hostname where the MongoDB database is running, defaults to "localhost".
    port : int, optional
        The port number to connect to the MongoDB database, defaults to 27017.
    username : str, optional
        The username for authenticating with the MongoDB database, defaults to an empty string.
    password : str, optional
        The password for authenticating with the MongoDB database, defaults to an empty string.
    verbose : bool, optional
        A flag to indicate whether to print verbose output, defaults to False.

    Attributes
    ----------
    uri : str
        The URI for connecting to the MongoDB database.
    client : pymongo.MongoClient
        The MongoDB client instance.
    db : pymongo.database.Database
        The Database instance from pymongo representing the MongoDB database.
    entries : pymongo.collection.Collection
        The Collection instance representing the entries in the MongoDB database.
    fs : dict
        A dictionary of GridFS instances for different collections in the MongoDB database.
    class_mapping : ClassMap
        The ClassMap instance for classifying the entries.
    db_info : dict
        A dictionary storing the information about the MongoDB database connection.
    """

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
        """
        .. _disconnect:

        Disconnect
        ==========

        Disconnects the Database object from the MongoDB database by closing the 
        pymongo client connection.

        Returns
        -------
        None
        """
        self.client.close()

    def __str__(self):
        """
        Return a string representation of the Database object.

        This method constructs a string that includes the MongoDB connection details 
        such as the database name, host, port, and user. The string is formatted 
        in a way that each detail is presented on a new line with the detail's name 
        and its value separated by a colon.

        Returns
        -------
        str
            A string representation of the Database object including the MongoDB 
            connection details.
        """
        ret = "MongoDB\n"
        ret += "----------------\n"
        for key in self.db_info:
            ret += "{:<15}: {}\n".format(key, self.db_info[key])
        return ret

    def add_collection(self, collection: str):
        """
        Add a new collection to the database.

        This method creates a new GridFS instance for the specified collection 
        and adds it to the `fs` attribute.

        Parameters
        ----------
        collection : str
            The name of the collection to add.

        Returns
        -------
        None
        """
        self.fs[collection] = gridfs.GridFS(self.db, collection)
        return

    def exists(self,
               filter: dict = None,
               collection: str = None,
               metadata: Metadata = None) -> Union[ObjectId, bool]:
        """
        Check if a document exists in the specified collection based on the provided filter or metadata.

        This method checks if a document exists in the specified collection of the MongoDB database 
        that matches the provided filter or metadata. If a document is found, it returns the ObjectId 
        of the document. If no document is found, it returns False.

        Parameters
        ----------
        filter : dict, optional
            A dictionary specifying the filter criteria to use when searching for the document.
        collection : str, optional
            The name of the collection to search in. If not provided, the collection name is 
            determined from the 'mode' field of the metadata.
        metadata : Metadata, optional
            A Metadata object specifying the metadata to use when searching for the document. 
            If provided, the 'mode' field of the metadata is used as the collection name and 
            the metadata is converted to a MongoDB filter.

        Returns
        -------
        ObjectId or bool
            The ObjectId of the found document if a matching document is found, False otherwise.

        Raises
        ------
        Exception
            If neither metadata nor filter and collection are provided.
        """
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
               obj: Union[Image, Tape, TapeAnalyzer],
               ext: str = '.png',
               overwrite: bool = False,
               skip: bool = False,
               collection: str = None) -> ObjectId:
        """
        Insert an object into the specified collection in the MongoDB database.

        This method inserts an object into the specified collection of the MongoDB database. 
        If the object already exists in the database, the behavior depends on the `overwrite` 
        and `skip` parameters. If `overwrite` is True, the existing object is deleted and the 
        new object is inserted. If `skip` is True, the insertion is skipped and the ObjectId 
        of the existing object is returned. If neither `overwrite` nor `skip` is True and the 
        object already exists, an exception is raised.

        Parameters
        ----------
        obj : Union[Image, Tape, TapeAnalyzer]
            The object to insert into the database.
        ext : str, optional
            The file extension to use when converting the object to a buffer, defaults to '.png'.
        overwrite : bool, optional
            Whether to overwrite the existing object if it already exists, defaults to False.
        skip : bool, optional
            Whether to skip the insertion if the object already exists, defaults to False.
        collection : str, optional
            The name of the collection to insert the object into. If not provided, the collection 
            name is determined from the 'mode' field of the object's metadata.

        Returns
        -------
        ObjectId
            The ObjectId of the inserted object.

        Raises
        ------
        Exception
            If the object already exists in the database and neither `overwrite` nor `skip` is True.
        """

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
                print(
                    f"{obj.metadata.filename} {collection} already exists, overwriting!")
            self.delete(filter, collection)
        elif skip and exists:
            if self.verbose:
                print(
                    f"{obj.metadata.filename} {collection} already exists, skipping!")
            return exists
        metadata = obj.metadata.to_serial_dict
        metadata['ext'] = ext
        filename = obj.metadata.filename
        _id = fs.put(obj.to_buffer(ext),
                     filename=filename,
                     metadata=metadata)
        return _id

    def find(self,
            filter: dict,
            collection: str = 'analysis',
            ext: str = '.png',
            version: int = -1,
            no_cursor_timeout: bool = False) -> list:
        """
        Find and return objects from the specified collection that match the provided filter.

        This method finds and returns objects from the specified collection of the MongoDB database 
        that match the provided filter. The objects are returned as instances of the class associated 
        with the collection in the class mapping. The objects are sorted by their upload date in the 
        order specified by the `version` parameter.

        Parameters
        ----------
        filter : dict
            A dictionary specifying the filter criteria to use when searching for the objects.
        collection : str, optional
            The name of the collection to search in, defaults to 'analysis'.
        ext : str, optional
            The file extension to use when converting the objects to buffers, defaults to '.png'.
        version : int, optional
            The sort order for the objects based on their upload date. If `version` is -1, the objects 
            are sorted in descending order. If `version` is 1, the objects are sorted in ascending order, 
            defaults to -1.
        no_cursor_timeout : bool, optional
            Whether to prevent the server-side cursor from timing out after an inactivity period, 
            defaults to False.

        Returns
        -------
        list
            A list of objects from the specified collection that match the provided filter. The objects 
            are returned as instances of the class associated with the collection in the class mapping.
        """
        Class = self.class_mapping[collection]
        fs = self.fs[collection]

        ret = []
        if self.count_documents(filter, collection) != 0:
            queries = fs.find(filter=filter,
                              no_cursor_timeout=no_cursor_timeout).sort("uploadDate", version)
            for iq in queries:
                ret.append(Class.from_buffer(iq.read(), iq.metadata))
        return ret

    def map_to(self,
            func: Callable,
            filter: dict,
            collection_source: str,
            collection_target: str,
            verbose: bool = True,
            no_cursor_timeout: bool = False):
        """
        Apply a function to each object in the source collection that matches the provided filter 
        and insert the results into the target collection.

        This method applies a function to each object in the source collection of the MongoDB database 
        that matches the provided filter. The results are inserted into the target collection. The 
        objects are retrieved as instances of the class associated with the source collection in the 
        class mapping.

        Parameters
        ----------
        func : Callable
            The function to apply to each object. The function should take an object as input and 
            return an object.
        filter : dict
            A dictionary specifying the filter criteria to use when searching for the objects in 
            the source collection.
        collection_source : str
            The name of the source collection to search in.
        collection_target : str
            The name of the target collection to insert the results into.
        verbose : bool, optional
            Whether to print the filename of each object being processed, defaults to True.
        no_cursor_timeout : bool, optional
            Whether to prevent the server-side cursor from timing out after an inactivity period, 
            defaults to False.

        Returns
        -------
        None
        """
        Class = self.class_mapping[collection_source]
        fs = self.fs[collection_source]

        queries = fs.find(filter=filter,
                          no_cursor_timeout=no_cursor_timeout)
        if queries is None:
            print('There are no matching entries to the provided filter')
        for iq in queries:
            obj = Class.from_buffer(iq.read(), iq.metadata)
            ext = obj.metadata['ext']
            if verbose:
                print(iq.filename)
            self.insert(func(obj), ext=ext, collection=collection_target)

    def find_one(self, filter: Optional[dict] = None, collection: Optional[str] = None) -> object:
        """
        Find and return one object from the specified collection that matches the provided filter.

        This method finds and returns one object from the specified collection of the MongoDB database 
        that matches the provided filter. The object is returned as an instance of the class associated 
        with the collection in the class mapping. If no collection is specified, one is chosen randomly.

        Parameters
        ----------
        filter : dict, optional
            A dictionary specifying the filter criteria to use when searching for the object. If not 
            provided, the first object in the collection is returned.
        collection : str, optional
            The name of the collection to search in. If not provided, a collection is chosen randomly.

        Returns
        -------
        object
            An object from the specified collection that matches the provided filter. The object is 
            returned as an instance of the class associated with the collection in the class mapping.

        Raises
        ------
        ValueError
            If no object is found that matches the provided filter.
        """
        if collection is None:
            collection = choice(list(self.fs))
        Class = self.class_mapping[collection]
        fs = self.fs[collection]
        iq = fs.find_one(filter)
        if iq is not None:
            return Class.from_buffer(iq.read(), iq.metadata)
        else:
            raise ValueError(f'No entry found with the filter: {str(filter)}')

    def find_with_id(self, _id: str, collection: str) -> object:
        """
        Find and return an object from the specified collection based on its MongoDB _id.

        This method finds and returns an object from the specified collection of the MongoDB database 
        based on its MongoDB _id. The object is returned as an instance of the class associated with 
        the collection in the class mapping.

        Parameters
        ----------
        _id : str
            The MongoDB _id of the object to find.
        collection : str
            The name of the collection to search in.

        Returns
        -------
        object
            An object from the specified collection with the provided MongoDB _id. The object is 
            returned as an instance of the class associated with the collection in the class mapping.
        """

        Class = self.class_mapping[collection]
        fs = self.fs[collection]
        iq = fs.find_one(
            {'_id': _id if type(_id) is ObjectId else ObjectId(_id)})
        return Class.from_buffer(iq.read(), iq.metadata)

    def filter_with_metadata(self,
                            inp: List[str],
                            filter: dict,
                            collection: str) -> List[int]:
        """
        Filter a list of filenames based on metadata and return the indices of the matching filenames.

        This method filters a list of filenames based on the provided filter and the metadata of the 
        files in the specified collection of the MongoDB database. It returns a list of indices of the 
        input list that correspond to the filenames that match the filter.

        Parameters
        ----------
        inp : List[str]
            The list of filenames to filter.
        filter : dict
            A dictionary specifying the filter criteria to use when filtering the filenames.
        collection : str
            The name of the collection to search in.

        Returns
        -------
        List[int]
            A list of indices of the input list that correspond to the filenames that match the filter.
        """
        fs = self.fs[collection]
        query = fs.find({'$and':
                        [
                            filter,
                            {'filename': {'$in': inp}}
                        ]
        })
        db_fnames = [iq.metadata['filename'] for iq in query]
        ret = []
        for i, filename in enumerate(inp):
            if filename in db_fnames:
                ret.append(i)
        return ret

    def count_documents(self, filter: dict, collection: str) -> int:
        """
        Count the number of documents in the specified collection that match the provided filter.

        This method counts the number of documents in the specified collection of the MongoDB database 
        that match the provided filter. If the collection does not exist, it returns 0.

        Parameters
        ----------
        filter : dict
            A dictionary specifying the filter criteria to use when counting the documents.
        collection : str
            The name of the collection to count the documents in.

        Returns
        -------
        int
            The number of documents in the specified collection that match the provided filter.
        """
        if collection not in self.fs:
            return 0
        fs = self.fs[collection]
        cursor = fs.find()
        return cursor.collection.count_documents(filter=filter)

    def export_to_files(self,
                        destination: str,
                        filter: dict,
                        collection: str,
                        ext: str = '.png',
                        verbose: bool = True,
                        no_cursor_timeout: bool = False):
        """
        Export objects from the specified collection that match the provided filter to files.

        This method exports objects from the specified collection of the MongoDB database that match 
        the provided filter to files. The objects are saved as files in the specified destination 
        directory. The objects are retrieved as instances of the class associated with the collection 
        in the class mapping.

        Parameters
        ----------
        destination : str
            The path to the directory where the files should be saved.
        filter : dict
            A dictionary specifying the filter criteria to use when searching for the objects.
        collection : str
            The name of the collection to search in.
        ext : str, optional
            The file extension to use when saving the objects, defaults to '.png'.
        verbose : bool, optional
            Whether to print the path of each file being saved, defaults to True.
        no_cursor_timeout : bool, optional
            Whether to prevent the server-side cursor from timing out after an inactivity period, 
            defaults to False.

        Returns
        -------
        None
        """
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
        """
        Drop the specified collection from the MongoDB database.

        This method drops the specified collection and its associated 'files' and 'chunks' collections 
        from the MongoDB database.

        Parameters
        ----------
        collection : str
            The name of the collection to drop.

        Returns
        -------
        None
        """
        for x in ['files', 'chunks']:
            print(f'drop {collection}.{x}')
            self.db.drop_collection(f'{collection}.{x}')
        return

    def delete(self, filter: dict, collection: str):
        """
        Delete documents from the specified collection that match the provided filter.

        This method deletes documents from the specified collection of the MongoDB database 
        that match the provided filter.

        Parameters
        ----------
        filter : dict
            A dictionary specifying the filter criteria to use when deleting the documents.
        collection : str
            The name of the collection to delete the documents from.

        Returns
        -------
        None
        """
        fs = self.fs[collection]
        queries = fs.find(filter)
        for iq in queries:
            fs.delete(iq._id)
        return

    def delete_database(self):
        """
        Delete the MongoDB database associated with this Database instance.

        This method deletes the MongoDB database that this Database instance is connected to.

        Returns
        -------
        None
        """
        self.client.drop_database(self.name)

    @property
    def collection_names(self) -> List[str]:
        """
        Get the names of the collections in the MongoDB database.

        This method returns the names of the collections in the MongoDB database that this Database 
        instance is connected to. The '.files' suffix is removed from the collection names.

        Returns
        -------
        List[str]
            A list of the names of the collections in the MongoDB database.
        """
        return [x.replace(".files", '')for x in self.db.list_collection_names() if 'files' in x]

    @property
    def connected(self) -> bool:
        """
        Check if the Database instance is currently connected to the MongoDB database.

        This method checks if the Database instance is currently connected to the MongoDB database 
        by attempting to retrieve the server information. If the server information is successfully 
        retrieved, the method returns True. If a ServerSelectionTimeoutError occurs, the method 
        prints the error and returns False.

        Returns
        -------
        bool
            True if the Database instance is currently connected to the MongoDB database, False otherwise.
        """
        try:
            self.client.server_info()  # force connection on a request as the
            return True
        except pymongo.errors.ServerSelectionTimeoutError as err:
            print(err)
            return False

    @property
    def server_info(self) -> dict:
        """
        Get the server information for the MongoDB database.

        This method retrieves and returns the server information for the MongoDB database that this 
        Database instance is connected to.

        Returns
        -------
        dict
            A dictionary containing the server information for the MongoDB database.
        """
        return self.client.server_info()


def dict2mongo_query(inp: dict, previous_key: str = '') -> dict:
    """
    Convert a dictionary into a MongoDB query.

    This function takes a dictionary and converts it into a MongoDB query. The keys of the 
    dictionary are concatenated with the `previous_key` parameter to form the keys of the query. 
    The values of the dictionary are used as the values of the query.

    Parameters
    ----------
    inp : dict
        The dictionary to convert into a MongoDB query.
    previous_key : str, optional
        The key to prepend to the keys of the dictionary when forming the keys of the query, 
        defaults to an empty string.

    Returns
    -------
    dict
        The MongoDB query formed from the input dictionary.
    """
    ret = []
    for key in inp:
        if isinstance(inp[key], dict):
            if len(inp[key]) != 0:
                ret.append(dict2mongo_query(inp[key],
                                            previous_key=previous_key+'.'+key,
                                            ))
            else:
                ret.append({previous_key+'.'+key: inp[key]})
        else:
            ret.append({previous_key+'.'+key: inp[key]})
    ret_p = []
    for item_1 in ret:
        if isinstance(item_1, list):
            for item_2 in item_1:
                ret_p.append(item_2)
        else:
            ret_p.append(item_1)
    return ret_p


def list_databases(host: str = 'localhost',
                   port: int = 27017,
                   username: str = '',
                   password: str = '') -> List[str]:
    """
    List the names of all databases on a MongoDB server.

    This function connects to a MongoDB server using the provided host, port, username, and password, 
    and returns a list of the names of all databases on the server.

    Parameters
    ----------
    host : str, optional
        The host IP address or hostname where the MongoDB server is running, defaults to 'localhost'.
    port : int, optional
        The port number to connect to the MongoDB server, defaults to 27017.
    username : str, optional
        The username for authenticating with the MongoDB server, defaults to an empty string.
    password : str, optional
        The password for authenticating with the MongoDB server, defaults to an empty string.

    Returns
    -------
    List[str]
        A list of the names of all databases on the MongoDB server.
    """
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
    """
    Dump a MongoDB database or collection to a BSON file.

    This function uses the `mongodump` command to dump a MongoDB database or collection to a BSON 
    file. The `mongodump` command is a utility for creating a binary export of the contents of a 
    database.

    Parameters
    ----------
    db : str, optional
        The name of the database to dump. If not provided, all databases are dumped.
    host : str, optional
        The host IP address or hostname where the MongoDB server is running. If not provided, 
        'localhost' is used.
    port : int, optional
        The port number to connect to the MongoDB server. If not provided, 27017 is used.
    username : str, optional
        The username for authenticating with the MongoDB server. If not provided, no authentication 
        is used.
    password : str, optional
        The password for authenticating with the MongoDB server. If not provided, no authentication 
        is used.
    out : str, optional
        The directory where the dump should be output. If not provided, the dump is output to the 
        'dump' directory in the current working directory.
    collection : str, optional
        The name of the collection to dump. If not provided, all collections in the specified 
        database are dumped.

    Returns
    -------
    None
    """
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
    """
    Restore a MongoDB database or collection from a BSON file.

    This function uses the `mongorestore` command to restore a MongoDB database or collection from 
    a BSON file. The `mongorestore` command is a utility for creating a binary import from the 
    contents of a BSON file.

    Parameters
    ----------
    path : str, optional
        The path to the BSON file. If not provided, the 'dump' directory in the current working 
        directory is used.
    db : str, optional
        The name of the database to restore. If not provided, all databases are restored.
    host : str, optional
        The host IP address or hostname where the MongoDB server is running. If not provided, 
        'localhost' is used.
    port : int, optional
        The port number to connect to the MongoDB server. If not provided, 27017 is used.
    username : str, optional
        The username for authenticating with the MongoDB server. If not provided, no authentication 
        is used.
    password : str, optional
        The password for authenticating with the MongoDB server. If not provided, no authentication 
        is used.
    collection : str, optional
        The name of the collection to restore. If not provided, all collections in the specified 
        database are restored.

    Returns
    -------
    None
    """
    command = 'mongorestore '
    tags = locals()
    for itag in tags:
        if tags[itag] is not None and itag not in ['command', 'path']:
            command += f'--{itag}={tags[itag]} '
    command += path
    print(command)
    subprocess.check_output(command, shell=True)
