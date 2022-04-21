# -*- coding: utf-8 -*-

import numpy as np
import io
from gridfs.grid_file import GridOut

def serializer(indict: dict) -> dict:
    """Serilizes any given dictionary for mongodb.

    Parameters
    ----------
    indict : dict
        input dictionary

    """
    
    ret = {}
    for key in indict:
        if type(indict[key]) is dict:
            ret[key] = serializer(indict[key])
        elif type(indict[key]) is np.ndarray :
            ret[key] = indict[key].tolist()
        else :
            ret[key] = indict[key]
    return ret

def read_bytes_io(obj: GridOut) -> np.array:
    """reads a binary file stored in mongodb and returns a numpy array

    Parameters
    ----------
    obj : GridOut
        output from a mongodb girdfs file

    Returns
    -------
    np.array
        numpy array containing the information loaded from gridfs file

    """
    
    return np.load(io.BytesIO(obj.read()), allow_pickle=True)


def write_bytes_io(obj: dict) -> io.BytesIO:
    output = io.BytesIO()
    np.savez(output, **obj)
    return output.getvalue()
    
