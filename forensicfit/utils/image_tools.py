# -*- coding: utf-8 -*-

import cv2
import numpy as np


def rotate_image(image, angle):
    """
    Rotates the image by angle degrees

    Parameters
    ----------
        angle : float
            Angle of rotation.

    Returns
    -------
        None.

    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    image = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return


def gaussian_blur(image, window=(15, 15)):
    """
    This method applies Gaussian Blur filter to the image. 

    Parameters
    ----------
    window : tuple int, optional
        The window in which the gaussian blur is going to be applied.
        The default is (15,15).

    Returns
    -------
    None.

    """
    image = cv2.GaussianBlur(image, window, 0)
    return image

def split_vertical(image, pixel_index=None,pick_side='L', flip=True):
    """
    This method splits the image in 2 images based on the fraction that is 
    given in pixel_index

    Parameters
    ----------
    pixel_index : float, optional
        fraction in which the image is going to be split. The value should
        be a number between zero and one. The default is 0.5.
    pick_side : str, optional
        The side in which will over write the image in the class. The 
        default is 'L'.

    Returns
    -------
    None.

    """
    width = image.shape[1]
    
    if pixel_index is None:
        pixel_index = width//2
    if pixel_index<1:
        pixel_index = int(width*pixel_index)
    if pick_side == 'L':
        image = image[:, 0:pixel_index]
    else : 
        if flip:
            image = cv2.flip(image[:,pixel_index:width],1)
        else:
            image = image[:, pixel_index:width]
    return image

def gray_scale(image):
    """
    Gray Scale image of the input image.

    Returns
    -------
    gray_scale : cv2 object
        Gray Scale image of the input image.

    """
    if len(image.shape) >2:
        gray_scale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else : 
        gray_scale = image
    return gray_scale

def flip(image):
    return cv2.flip(image, 0)
    
def contours(image, mask_threshold=60):
    """
     A list of pixels that create the contours in the image

    Returns
    -------
    contours : list 
        A list of pixels that create the contours in the image

    """
    masked = get_masked(image, mask_threshold)
    contours,_ = cv2.findContours(
        masked,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    return contours


def largest_contour(contours):
    """
    A list of pixels forming the contour with the largest area

    Returns
    -------
    contour_max_area : list
        A list of pixels forming the contour with the largest area

    """
    
    max_con = 0
    max_con_arg = 0
    for ic in range(len(contours)):
        if cv2.contourArea(contours[ic]) > max_con :
            max_con = cv2.contourArea(contours[ic])
            max_con_arg = ic
    contour_max_area = contours[max_con_arg]
    return contour_max_area

def get_masked(image, mask_threshold):
        """
        Populates the masked image with the gray scale threshold
        Returns
        
        -------
        None.

        """
        masked = cv2.inRange(image,
                             mask_threshold,
                             255)
        
        return masked
    
def resize(image, size):
    """
    This method resizes the image to the pixel size given.

    Parameters
    ----------
    size : tuple int,
        The target size in which the image is going to be resized. 

    Returns
    -------
    None.

    """

    image = cv2.resize(image,size)
    return image
    
    

def binerized_mask(image, masked):
    """
    This funtion return the binarized version of the tape

    Returns
    -------
    2d array of the image
        .

    """
    image = cv2.bitwise_and(image,
                           image,
                           mask=masked)
    return cv2.bitwise_and(image,
                           image,
                           mask=masked)