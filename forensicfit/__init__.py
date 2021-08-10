# -*- coding: utf-8 -*-

try:
    import cv2
    has_opencv = True
except:
    has_opencv = False

try:
    import pymongo
    has_pymongo = True
except:
    has_pymongo = False
from . import utils
from . import core
if has_opencv and has_pymongo:
    from . import database
    from .script_process_dir import process_directory
    from .store_on_db import store_on_db
    from .from_excel import from_excel

