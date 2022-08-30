# -*- coding: utf-8 -*-
import importlib.util
import sys
def has_package(name):
    spec = importlib.util.find_spec(name)
    if name in sys.modules:
        ret = True
    elif spec is not None:
        ret = True
    else:
        ret = False
    return ret



HAS_OPENCV = has_package("cv2")
HAS_SKIMAGE = has_package("skimage")
HAS_PYMONGO = has_package("pymongo")
HAS_TENSORFLOW = has_package("tensorflow")


from . import core
from . import utils
if HAS_TENSORFLOW:
    from . import machine_learning
if HAS_OPENCV and HAS_PYMONGO:
    from . import database as db
    # from .script_process_dir import process_directory
    # from .store_on_db import store_on_db
    # from .from_excel import from_excel

