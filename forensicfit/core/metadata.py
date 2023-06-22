"""
metadata.py
===========
This module contains the Metadata class which is used to manage the metadata of images for forensic analysis.

The Metadata class implements a MutableMapping, which provides a dict-like interface. This class offers methods 
for setting and retrieving metadata values, as well as special conversion methods that adapt the metadata to various
needs, such as MongoDB filter creation or serialization for storage in MongoDB.

The Metadata class makes use of the serializer function from the `array_tools` utility module, which converts nested 
dictionaries and numpy arrays to a suitable form for storage in MongoDB.

This module is part of the 'forensicfit' package which aims to provide tools for forensic analysis of images.

Author: Pedram Tavadze
Email: petavazohi@gmail.com
"""

__all__ = []
__version__ = '1.0'
__author__ = 'Pedram Tavadze'

from abc import ABCMeta, abstractmethod
from collections.abc import MutableMapping
from pathlib import Path
from ..utils.array_tools import serializer
from typing import Any


class Metadata(MutableMapping):
    """
    Metadata is a class that extends from MutableMapping to provide dictionary-like functionality.
    It holds and manipulates arbitrary metadata attached to its instances.

    It is meant to be used for manipulating metadata of images and implements the MutableMapping 
    abstract base class.

    Parameters
    ----------
    inp : dict, optional
        Initial metadata dictionary, by default None

    **kwargs
        Additional metadata passed as keyword arguments, by default None
    """
    
    def __init__(self, inp: dict = None, **kwargs):
        
        self.mapping = {}
        self.update(inp)

    def __contains__(self, x: str) -> bool:
        """
        Checks if a given key exists in the metadata.

        The method overrides the built-in Python `__contains__` to provide a custom implementation 
        for the Metadata class.

        Parameters
        ----------
        x : str
            The key to check for existence in the metadata dictionary.

        Returns
        -------
        bool
            True if the key exists in the metadata, False otherwise.
        """
        return x in self.mapping

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Assign a value to a specified key in the metadata.

        This method overrides the built-in Python `__setitem__` method for dictionary 
        objects to provide a custom implementation for the Metadata class. It also checks 
        if the provided value is a pathlib.Path instance and converts it to a string 
        before assigning it to the metadata. If the key does not correspond to an existing 
        class property, the method also sets an instance attribute with the same name.

        Parameters
        ----------
        key : str
            The key for which to assign the value.

        value : Any
            The value to assign to the key. If it is a pathlib.Path instance, 
            it's converted to a string.

        Returns
        -------
        None
        """
        if 'pathlib' in str(type(value)):
            value = value.as_posix()
        self.mapping[key] = value
        if not isinstance(getattr(type(self), key, None), property):
                setattr(self, key, value)
        return 

    def __delitem__(self, key):
        del self.mapping[key]

    def __getitem__(self, x):
        return self.mapping.__getitem__(x)

    def __iter__(self):
        return self.mapping.__iter__()

    def __len__(self):
        return self.mapping.__len__()

    def __repr__(self):
        """
        Generate a string representation of the Metadata object.

        This method overrides the built-in Python `__repr__` method to provide a custom 
        string representation of the Metadata object. This representation excludes 
        the 'boundary' and 'analysis' keys and includes a descriptive version 
        of each key and its corresponding value, making it more human-readable.

        Returns
        -------
        str
            The string representation of the Metadata object.
        """
        ret = ''
        for key in self.mapping:
            if key not in ['boundary', 'analysis']:
                value = self.mapping[key] 
                desc = key.capitalize() if key != 'dpi' else key.upper()
                desc = desc.replace('_', ' ')
                desc = desc.replace(' h', ' horizontal')
                desc = desc.replace(' v', ' vertical')
                ret += f'{desc}: {value}\n'
        return ret
    
   
    def __add__(self, new):
        for key in new.mapping:
            if key not in self:
                self[key] = new[key]
        return self
    
    def to_mongodb_filter(self, inp: dict = None, previous_key: str = 'metadata') -> dict:
        """
        Converts the metadata object into a MongoDB filter query.

        This method is used to generate a list of key-value pairs suitable for MongoDB 
        filter operations. The keys are composed of the provided previous_key 
        and the subsequent nested keys (if any) in the Metadata object.

        Parameters
        ----------
        inp : dict, optional
            Input dictionary to be converted, if not provided the serialized form 
            of the current metadata object is used.
        previous_key : str, optional
            The initial key string that precedes all others in the final MongoDB filter 
            representation, by default 'metadata'.

        Returns
        -------
        list
            A list of dictionaries where each dictionary is a key-value pair
            suitable for use in a MongoDB filter operation.
        """
        inp = inp or self.to_serial_dict
        ret = []
        for key in inp:
            if  isinstance(inp[key], dict) :
                if len(inp[key]) != 0 :
                    ret.append(self.to_mongodb_filter(inp[key],
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
    
    @property
    def to_serial_dict(self) -> dict:
        """
        Returns the serialized form of the metadata dictionary.

        This method applies the `serializer` function to the metadata dictionary. 
        The serialization involves conversion of any nested dictionary or numpy array 
        in the metadata to a form suitable for storage in MongoDB.

        Returns
        -------
        dict
            The serialized form of the metadata dictionary. Any numpy arrays in the 
            dictionary have been converted to list format.
        """
        return serializer(self.mapping)
