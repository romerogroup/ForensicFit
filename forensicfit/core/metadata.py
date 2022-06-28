__all__ = []
__version__ = '1.0'
__author__ = 'Pedram Tavadze'

# built-ins
from abc import ABCMeta, abstractmethod
from collections.abc import MutableMapping
from pathlib import Path
from ..utils.array_tools import serializer


class Metadata(MutableMapping):
    # __metaclass__ = ABCMeta
    
    def __init__(self, inp: dict = None, **kwargs):
        
        self.mapping = {}
        self.update(inp)

    def __contains__(self, x):
        return x in self.mapping

    def __setitem__(self, key, value) -> None:
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

    def __str__(self):
        ret = ''
        for key in self.mapping:
            value = self.mapping[key]
            ret += f'{key} : {value}\n'
        return ret
    
    def __add__(self, new):
        for key in new.mapping:
            if key not in self:
                self[key] = new[key]
        return self
    
    def to_mongodb_filter(self, inp: dict = None, previous_key: str = 'metadata') -> dict:
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
    def to_serial_dict(self):
        return serializer(self.mapping)
