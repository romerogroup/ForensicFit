# -*- coding: utf-8 -*-

import io

import cv2
import numpy as np
from numpy import typing as npt

from .. import HAS_PYMONGO

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
        elif type(indict[key]) is np.ndarray:
            ret[key] = indict[key].tolist()
        else :
            ret[key] = indict[key]
    return ret
    
def vote_calculator(prediction: npt.ArrayLike) -> npt.ArrayLike:
    n_voters = prediction.shape[1]
    ret = np.zeros(shape=(prediction.shape[0]))
    for i, pred in enumerate(prediction):
        votes = pred.round()
        score = votes.sum()/n_voters
        ret[i] = score
    return np.array(ret)


if HAS_PYMONGO:
    from gridfs.grid_file import GridOut


    def read_bytes_io(obj: GridOut, method: str = 'numpy') -> np.array:
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
        if method == 'numpy':
            return np.load(io.BytesIO(obj.read()), allow_pickle=True)
        elif method == 'opencv':
            return cv2.imdecode(np.frombuffer(obj.read().getbuffer(), np.uint8), -1)


    def write_bytes_io(obj: dict, method: str = 'numpy') -> io.BytesIO:
        if method == 'numpy':
            output = io.BytesIO()
            np.savez(output, **obj)
            return output.getvalue()
        elif method == 'opencv':
            is_success, buffer = cv2.imencode(".png", obj)
            output = io.BytesIO(buffer)
            return output.getvalue()

