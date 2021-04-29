# -*- coding: utf-8 -*-

import numpy as np

def serializer(indict):
    for key in indict:
        if type(indict[key]) is dict:
            indict[key] = serializer(indict[key])
        elif type(indict[key]) is np.ndarray :
            indict[key] = indict[key].tolist()
    return indict
            