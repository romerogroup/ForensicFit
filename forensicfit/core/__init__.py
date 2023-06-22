# -*- coding: utf-8 -*-
"""
.. _core_module:

Core Module
===========

The core subpackage provides fundamental classes and functions needed for the operation of the ForensicFit package.

Dependencies
------------
The core subpackage requires the following external libraries:

- OpenCV (``HAS_OPENCV``)
- PyMongo (``HAS_PYMONGO``)

If these libraries are not installed, relevant functionalities might be disabled.

Main Classes
------------
The core subpackage defines the following primary classes:

- :class:`.Image` : Class for handling and manipulating images.
- :class:`.Metadata` : Class for handling metadata associated with images.
- :class:`.Analyzer` : Base class for implementing different types of image analysis.
- :class:`.Tape` : Class for creating and manipulating tape objects.
- :class:`.TapeAnalyzer` : Class that extends Analyzer, specialized in analyzing tape images.

Note
----
These classes form the backbone of the ForensicFit package and are used throughout its various functions and methods.

"""

from .. import HAS_OPENCV, HAS_PYMONGO
from .image import Image
from .metadata import Metadata
from .analyzer import Analyzer
from .tape import Tape, TapeAnalyzer

