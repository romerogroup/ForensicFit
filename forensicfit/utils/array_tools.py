# -*- coding: utf-8 -*-

import numpy as np

def serializer(indict):
    ret = {}
    for key in indict:
        if type(indict[key]) is dict:
            ret[key] = serializer(indict[key])
        elif type(indict[key]) is np.ndarray :
            ret[key] = indict[key].tolist()
        else :
            ret[key] = indict[key]
    return ret
            