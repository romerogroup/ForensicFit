# -*- coding: utf-8 -*-

from skimage.color import rgb2gray
import cv2
import numpy as np
import matplotlib.pylab as plt
from skimage import exposure, filters
from pathlib import Path


IMAGE_EXTENSIONS = ['.png', '.bmp', '.dib', '.jpeg', 
                    '.jpg', '.jpe', '.jp2', '.webp',
                    '.pbm', '.pgm', '.ppm', '.pxm', 
                    '.pnm', '.sr', '.ras', '.tiff',
                    '.tif', '.exr', '.hdr', '.pic']


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
    return image


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

def split_v(image, pixel_index=None,pick_side='L', flip=True):
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
            image = np.fliplr(image[:,pixel_index:width])
        else:
            image = image[:, pixel_index:width]
    return image

def to_gray(image: np.ndarray, mode='SD') -> np.ndarray:
    """Gray Scale image of the input image.
    
    modes: 'BT.470' and 'BT.709'
    SD 'BT.470' : Y = 0.299 R + 0.587 G + 0.114 B
    HD 'BT.709' : Y = 0.2125 R + 0.7154 G + 0.0721 B

    Returns
    -------
    gray_scale : cv2 object
        Gray Scale image of the input image.

    """
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3:
        if image.shape[2] == 1:
            return image
        elif image.shape[2] == 3:
            if mode == 'SD':
                return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            elif mode == 'HD':
                return rgb2gray(image)
            

def to_rbg(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        return image
    else:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

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

def remove_background(image: np.array, 
                      contour: np.array, 
                      outside: bool = True,
                      pixel_value: int = 0):
    """Removes the background outside or inside the contour

    Parameters
    ----------
    image : np.array
        _description_
    contour : np.array
        _description_
    outside : bool, optional
        _description_, by default True
    pixel_value : int, optional
        _description_, by default 0
    """
    bkg = np.ones_like(image)*pixel_value
    bkg = cv2.fillPoly(bkg, [contour], (255, 255, 255))

    image = cv2.bitwise_and(image,
                        bkg)
                        # mask=bkg)
    return image

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
    size = tuple(size)
    assert len(size) == 2, 'Resizing needs two dimensions'
    image = cv2.resize(image,size)
    return image

def exposure_control(image: np.ndarray, 
                        mode:str='equalize_hist', 
                        **kwargs) -> np.ndarray:
    """modifies the exposure

    Parameters
    ----------
    mode : str, optional
        Type of exposure correction. It can be selected from the options:
        ``'equalize_hist'`` or ``'equalize_adapthist'``. 
        `equalize_hist <https://scikit-image.org/docs/stable/api/skimage.exposure.html#equalize-hist>`_ 
        and `equalize_adapthist <https://scikit-image.org/docs/stable/api/skimage.exposure.html#equalize-adapthist>`
        use sk-image. by default 'equalize_hist'
    """
    exps = {'equalize_hist':exposure.equalize_hist,
            'equalize_adapthist':exposure.equalize_adapthist}
    assert mode in exps, 'Mode not valid.'
    image = exps[mode](image, **kwargs)
    return image

def apply_filter(image: np.ndarray, mode:str, **kwargs) -> np.ndarray:
    """Applies different types of filters to the image

    Parameters
    ----------
    mode : str
        Type of filter to be applied. The options are
        * ``'meijering'``: <Meijering neuriteness filter https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.meijering>_,
        * ``'frangi'``: < Frangi vesselness filter https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.frangi>_,
        * ``'prewitt'``: <Prewitt transform https://scikit-image.org/docs/stable/api/skimage.filters.html#prewitt>_,
        * ``'sobel'``: <Sobel filter https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel>_,
        * ``'scharr'``: <Scharr transform https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr>,
        * ``'roberts'``: <Roberts' Cross operator https://scikit-image.org/docs/stable/api/skimage.filters.html#examples-using-skimage-filters-roberts>_,
        * ``'sato'``: <Sato tubeness filter https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sato>_.
    """
    flts = {
        'meijering':filters.meijering,
        'frangi': filters.frangi,
        'prewitt': filters.prewitt,
        'sobel': filters.sobel,
        'scharr': filters.scharr,
        'roberts': filters.roberts,
        'sato': filters.sato
    }
    assert mode in flts, 'Filter not valid.'
    if mode == 'roberts':
        if image.ndim != 2:
            print('Cannot apply roberts to color images')
            return
    image = flts[mode](image, **kwargs)
    return image


def binerized_mask(image, masked):
    """
    This function return the binarized version of the tape

    Returns
    -------
    2d array of the image
        .

    """
    # image = cv2.bitwise_and(image,
    #                        image,
    #                        mask=masked)
    return cv2.bitwise_and(image,
                           image,
                           mask=masked)
    
def imwrite(fname: str, image: np.array):
    """save any 2d numpy array (or list) to an image file

    Parameters
    ----------
    fname : str
        flie name to be saved
    image : np.array
        2d numpy array (or list) to be saved
    """
    fname = Path(fname)
    fname.parent.mkdir(exist_ok=True)
    fname = fname.as_posix()
    if HAS_OPENCV:
        cv2.imwrite(fname, image)
    else:
        plt.imsave(fname, image)
