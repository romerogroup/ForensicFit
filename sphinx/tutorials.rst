Tutorials
=========

The following set of tutorials explains the basic usage of this package. 
The examples provide information on how to use the preprocessing method for different
approaches. Later it will explain how one can use the data generated in this 
section in different Machine Learning approaches using keras package.

Preprocessing
-------------

At first we will explain the steps one has to take to process a single image, 
Then we will move to processing directories of images. Each gray scale image is constructed from a grid of pixels. Each pixel contains a number from 0 to 255 representing the gray shade of that specific pixel.

Single image processing
~~~~~~~~~~~~~~~~~~~~~~~

All of our tutorials are eplained in ipython interactive session, However these
commands can be easily moved to a python script for more automation.


Loading the image
^^^^^^^^^^^^^^^^^
Given the path to the image forensicfit will create a TapeImage object.
Usage::

  import forensicfit
  tape_image = forensicfit.TapeImage(fname='L001.tiff')

``tape_image`` is a TapeImage object with different properties and method in this section we will explain the usage of each method and property.
Before explaining each of the methods, let's list the input parameter of this class. Any of the following parameters can be used as an input for TapeMatch by putting tha parameter between the parentheses.  


        fname :
	    fname defines the path to the image file. This can be any format the `opencv <https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html>`_ supports.

	    example: ``TapeImage(fname='L001.tiff')``
        tape_label :
            Tape label is used to label the processed image.

	    example: ``TapeImage(fname='L001.tiff',tape_label='LQ_R')``

	mask_threshold : 
	    Inorder to find the boundaries of the tape, the algorithm changes every pixel with a value lower than mask_threshold to 0(black)

	    default: ``mask_threshold=60``
        rescale : float, optional

	    Only for scale images down to a smaller size for example 
            rescale=1/2. The default is None.
        split : bool, optional
            Whether or not to split the image. The default is True.
        gaussian_blur : 2d tuple of int, optional
            Defines the window in which Gaussian Blur filter is applied. The 
            default is (15,15).
        split_side : string, optional
            After splitting the image which side is chosen. The default is 'L'.
        split_position : float, optional
            Number between 0-1. Defines the where the vertical split is going 
            to happen. 1/2 will be in the middle. The default is None.
         
    


Show
^^^^

.. toctree::
   
   preprocess
   preprocess_dir
   machine_learning
   
