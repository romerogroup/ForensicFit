# -*- coding: utf-8 -*-
from .. import HAS_OPENCV
from .. import HAS_PYMONGO
if HAS_PYMONGO:
    from . import array_tools
if HAS_OPENCV:
    from . import image_tools
from . import plotter
from .general import copy_doc


import os
import json
from glob import glob
import logging.config
from pathlib import Path


# Other Constants
FILE = Path(__file__).resolve()
ROOT = str(FILE.parents[1])  # forensicfit
