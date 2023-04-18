.. Forensics documentation master file, created by
   sphinx-quickstart on Wed Aug  5 21:43:15 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ForensicFit Documentation!
-------------------------------------
ForensicFit is a Python package designed to preprocess scanned images from various sources and generate a database to be used in different machine learning approaches. This package prepares the data using four distinct techniques, which will be explained in the tutorial sections. ForensicFit leverages state-of-the-art image processing methods to analyze and store the generated data, ensuring compatibility with popular machine learning packages such as TensorFlow, PyTorch, and SciKit-learn. It utilizes NumPy, SciPy, matplotlib, OpenCV, scikit-image, PyMongo, and GridFS. For ease of use and future development, the package adheres to PEP-257 and PEP-484 for documentation and type hints, respectively.

Package Structure
-----------------

ForensicFit is organized into three main sub-packages: ``core``, ``database``, and ``utils``.

* ``core``: This sub-package contains the essential functionalities of ForensicFit, including Python classes that manage read/write, analysis, and metadata storage. These classes provide a data structure skeleton for the package and define standards for future implementations related to different types of materials.
* ``database``: This sub-package offers an efficient and flexible method for storing and retrieving raw and preprocessed data. Although the rest of the package does not depend on this sub-package, it has been included to simplify the data storage and query process. Users can still store and access raw or analyzed data using traditional image storage methods.
* ``utils``: The ``utils`` sub-package contains various image manipulation, plotting, and memory access tools used throughout the package.

.. image:: images/ForensicFit_tree.svg
    :align: center
    

Installation
------------
To install ForensicFit, use the following command:

.. code-block:: bash

      pip install forensicfit

Quick Start
-----------
Here's a quick example of how to use ForensicFit:

.. code-block:: python

      >>> import forensicfit as ff

      # Load your image file
      >>> path = 'path/to/LQ-HT-1.jpg'
      >>> tape = ff.core.Tape.from_file(path)
      >>> print(tape)
      Mode: material
      Resolution: (2471, 6289, 3)
      Path: path/to/LQ-HT-1.jpg/LQ_099.tif
      Filename: LQ_099.tif
      Compression: raw
      DPI: (1200.0, 1200.0)
      Flip horizontal: False
      Flip vertical: False
      Split vertical: {'side': None, 'pixel_index': None}
      Label: None
      Material: tape
      Surface: None
      Stretched: False

.. image:: images/LQ_099-raw.png
   :align: center



.. toctree::
   :maxdepth: 4
   :caption: Contents:

   installation
   developers
   tutorials

   modules  

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

