Tutorials
=========

The following set of tutorials explains the basic usage of this package. 
The examples provide information on how to use the preprocessing method for different
approaches. Later it will explain how one can use the data generated in this 
section in different Machine Learning approaches using keras package.

Preprocessing
-------------

At first we will explain the steps one has to take to process a single image, 
Then we will move to processing directories of images. 

Single image processing
~~~~~~~~~~~~~~~~~~~~~~~

All of our tutorials are eplained in ipython interactive session, However these
commands can be easily moved to a python script for more automation.

^^^^^^^^^^^^^^^^^
Loading the image
^^^^^^^^^^^^^^^^^

Usage::

  import edge_matching
  tape_image = edge_matching.TapeImage('L001.tiff')
    



.. toctree::
   
   preprocess
   preprocess_dir
   machine_learning
   
