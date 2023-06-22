# -*- Coding: utf-8 -*-
"""
tape.py

This module contains classes for handling and analyzing measurements 
from tape in the context of the ForensicFit application. These 
measurements are represented as images which are processed and 
analyzed using a variety of computer vision techniques.

The module includes the following classes:

- Tape: A class that represents an image of a tape measurement. It 
provides functionalities to load and process the image, extract 
relevant information, and perform various operations such as 
binarization and smearing.

- TapeAnalyzer: A class that inherits from the Tape class, adding 
analysis functionalities. It can calculate the tape's boundary, 
plot it, and compute and store analysis metadata.

The Tape class represents a single tape measurement and provides 
basic image processing operations. The TapeAnalyzer class extends 
this functionality by adding methods to analyze the tape's boundary 
and other features, which can be useful in forensic applications.

Author: Pedram Tavadze
Email: petavazohi@gmail.com
"""
import math
from pathlib import Path
from typing import Union, Tuple, List, Dict, Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..utils import image_tools
from . import Image, Analyzer, HAS_OPENCV

if not HAS_OPENCV:
    print("To enable analyzer please install opencv")

class Tape(Image):
    """Tape class is used for preprocessing tape images for machine learning.
    
    This class takes in a tape image, detects the edges, auto crops the image and
    returns the results in 3 different methods: coordinate_based, bin_based,
    and max_contrast.
    
    Parameters
    ----------
    image : np.ndarray
        The image to be processed. It must be a 2D numpy array.
    label : str, optional
        The label associated with the tape, by default None.
    surface : str, optional
        The surface the tape is on, by default None.
    stretched : bool, optional
        Flag indicating whether the tape is stretched or not, by default False.
    **kwargs
        Arbitrary keyword arguments.
    
    Attributes
    ----------
    metadata : dict
        A dictionary storing metadata of the image such as flipping information,
        splitting information, label, material, surface, stretched status, and mode.
    
    Raises
    ------
    AssertionError
        If `image` is not a numpy array or not a 2D array.
    """
    def __init__(self,
                 image: np.ndarray,
                 label: str = None,
                 surface: str = None,
                 stretched: bool=False,
                 **kwargs):
        assert type(image) is np.ndarray, "image arg must be a numpy array"
        assert image.ndim in [2, 3] , "image array must be 2 dimensional"
        super().__init__(image, **kwargs)
        self.metadata['flip_h'] = False
        self.metadata['flip_v'] = False
        self.metadata['split_v'] = {
            "side": None, "pixel_index": None}
        self.metadata['label'] = label
        self.metadata['material'] = 'tape'
        self.metadata['surface'] = surface
        self.metadata['stretched'] = stretched
        self.metadata['mode'] = 'material'
        
    def split_v(self, side: str, correct_tilt: bool=True, pixel_index: int=None):
        """Splits the tape image vertically.
        
        Parameters
        ----------
        side : str
            The side of the image to be processed.
        correct_tilt : bool, optional
            If set to True, tilt correction is applied on the image, by default True.
        pixel_index : int, optional
            The index of the pixel where the image is to be split, by default None.
        
        Returns
        -------
        None
        
        Notes
        -----
        Changes the 'image', 'split_v', 'image_tilt', 'tilt_corrected', 'resolution' fields of the metadata attribute.
        """
        self.values['original_image'] = self.image.copy()
        if pixel_index is None:
            tape_analyzer = TapeAnalyzer(self, correct_tilt=correct_tilt, auto_crop=True)
            x = tape_analyzer.boundary[:, 0]
            pixel_index = int((x.max()-x.min())/2+x.min())
            self.image = tape_analyzer['image']
            self.metadata['image_tilt'] = tape_analyzer.metadata.image_tilt
            self.metadata['tilt_corrected'] = tape_analyzer.metadata.tilt_corrected
        self.image = image_tools.split_v(
            self.image, pixel_index, side)
        self.values['image'] = self.image
        self.metadata['split_v'] = {
            "side": side, "pixel_index": pixel_index}
        self.metadata['resolution'] = self.shape
        



class TapeAnalyzer(Analyzer):
    """
    The TapeAnalyzer class is a specialized Analyzer used to preprocess duct tape images.

    This class is used to process images of duct tape to prepare them for machine learning tasks.
    It includes functionality for Gaussian blur, auto-cropping, tilt correction, and background removal.

    Parameters
    ----------
    tape : Tape, optional
        An instance of the Tape class representing the tape image to be analyzed.
    mask_threshold : int, optional
        The threshold used for masking during the image preprocessing. The default is 60.
    gaussian_blur : tuple, optional
        The kernel size for the Gaussian blur applied during the preprocessing. The default is (15, 15).
    n_divisions : int, optional
        The number of divisions to be used in the analysis. The default is 6.
    auto_crop : bool, optional
        A flag indicating whether the image should be auto-cropped. The default is False.
    correct_tilt : bool, optional
        A flag indicating whether the image tilt should be corrected. The default is False.
    padding : str, optional
        The padding method used in the analysis. The default is 'tape'.
    remove_background : bool, optional
        A flag indicating whether the background should be removed from the image. The default is True.
    """
    def __init__(self,
                 tape: Tape,
                 mask_threshold: int=60,
                 gaussian_blur: tuple=(15, 15),
                 n_divisions: int=6,
                 auto_crop: bool=False,
                 correct_tilt: bool=False,
                 padding: str='tape',
                 remove_background: bool=True):
        super().__init__()
        self.metadata['n_divisions'] = n_divisions
        self.metadata['gaussian_blur'] = gaussian_blur
        self.metadata['mask_threshold'] = int(mask_threshold)
        self.metadata['remove_background'] = remove_background
        self.metadata['padding'] = padding
        self.metadata["analysis"] = {}
        if tape is not None:
            self.image = tape.image
            self.metadata += tape.metadata
            self.metadata['resolution'] = self.image.shape
            self.preprocess()
            self.metadata['cropped'] = auto_crop
            self.metadata['tilt_corrected'] = correct_tilt
            if correct_tilt:
                angle = self.get_image_tilt()
                self.image = image_tools.rotate_image(self.image, angle)
                self.metadata.tilt_corrected = True
            if auto_crop:
                self.get_image_tilt()
                self.auto_crop_y()
                self.metadata.cropped = True
            if correct_tilt or auto_crop:
                self.preprocess()
            if remove_background:
                self.image = self['masked']
        return

    def preprocess(self):
        """
        The preprocess method applies various image processing techniques to prepare the image for analysis.

        This method performs several image preprocessing tasks including color inversion if necessary, conversion to grayscale, Gaussian blur, contour detection, and retrieval of the largest contour.

        Parameters
        ----------
        calculate_tilt : bool, optional
            A flag indicating whether the image tilt should be calculated. The default is True.
        auto_crop : bool, optional
            A flag indicating whether the image should be auto-cropped. The default is True.

        Returns
        -------
        None.

        Notes
        -----
        The original image is processed and the largest contour of the processed image is stored in the metadata under the 'boundary' key.
        """
        image = self.image
        if np.average(image) > 200:
            image = 255 - self.image
        # if self.metadata.gaussian_blur is not None:
        #     image = image_tools.gaussian_blur(
        #         self.image, window=self.metadata.gaussian_blur)
        gray = image_tools.to_gray(image, mode='SD')
        image = image_tools.gaussian_blur(
                gray, window=self.metadata.gaussian_blur)
        contours = image_tools.contours(image,
                                        self.metadata.mask_threshold)
        largest_contour = image_tools.largest_contour(contours)
        self.metadata['boundary'] = largest_contour.reshape(-1, 2)
            
    def flip_v(self):
        """
        Flips the image vertically and updates the relevant metadata.

        The flip_v method flips the image vertically, i.e., around the y-axis. It also updates the associated 
        metadata such as the boundary, coordinates, slopes and dynamic positions if they are present in the metadata.
        The 'flip_v' metadata attribute is also toggled.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        The original image is flipped vertically, and the corresponding changes are reflected in the metadata. The 'flip_v'
        metadata attribute is also toggled to reflect whether a vertical flip has been performed.
        """
        self.image = np.fliplr(self.image)
        self.metadata['boundary'] = np.array(self.metadata['boundary'])
        self.metadata['boundary'][:, 0] = self.metadata['boundary'][:, 0]*-1 + self.image.shape[1]
        if 'coordinate_based' in self.metadata['analysis']:
            coords  = np.array(self.metadata['analysis']['coordinate_based']['coordinates'])
            slopes  = np.array(self.metadata['analysis']['coordinate_based']['slopes'])
            coords[:, 0] *= -1
            coords[:, 0] += self.image.shape[1]
            slopes[:, 1] += slopes[:, 0]*self.image.shape[1]
            slopes[:, 0] *= -1
            # slopes[:, 1] += self.image.shape[1]
            self.metadata['analysis']['coordinate_based']['coordinates'] = coords
            self.metadata['analysis']['coordinate_based']['slopes'] = slopes
        if 'bin_based' in self.metadata['analysis']:
            dynamic_positions = np.array(self.metadata['analysis']['bin_based']['dynamic_positions'])
            for i, pos in enumerate(dynamic_positions):
                for ix in range(2):
                    dynamic_positions[i][0][ix] = pos[0][ix]*-1 + self.image.shape[1]
            dynamic_positions[:, 0, [0, 1]] = dynamic_positions[:, 0, [1, 0]]
            
            self.metadata['analysis']['bin_based']['dynamic_positions'] = dynamic_positions
        self.metadata['flip_v'] =  not(self.metadata['flip_v'])

    def flip_h(self):
        """
        Flips the image horizontally and updates the relevant metadata.

        The flip_h method flips the image horizontally, i.e., around the x-axis. It also updates the associated 
        metadata such as the boundary, coordinates, slopes and dynamic positions if they are present in the metadata.
        The 'flip_h' metadata attribute is also toggled.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        The original image is flipped horizontally, and the corresponding changes are reflected in the metadata. The 'flip_h'
        metadata attribute is also toggled to reflect whether a horizontal flip has been performed.
        """
        self.image = np.flipud(self.image)
        self.metadata['boundary'] = np.array(self.metadata['boundary'])
        self.metadata['boundary'][:, 1] = self.metadata['boundary'][:, 1]*-1 + self.image.shape[0]
        if 'coordinate_based' in self.metadata['analysis']:
            coords = np.array(self.metadata['analysis']['coordinate_based']['coordinates'])
            slopes  = np.array(self.metadata['analysis']['coordinate_based']['slopes'])
            coords[:, 1] *= -1
            coords[:, 1] += self.image.shape[0]
            slopes *= -1
            slopes[:, 1] += self.image.shape[0]
            self.metadata['analysis']['coordinate_based']['coordinates'] = coords
            self.metadata['analysis']['coordinate_based']['slopes'] = slopes
        if 'bin_based' in self.metadata['analysis']:
            dynamic_positions = np.array(self.metadata['analysis']['bin_based']['dynamic_positions'])
            for i, pos in enumerate(dynamic_positions):
                for ix in range(2):
                    dynamic_positions[i, 1, ix] = pos[1, ix]*-1 + self.image.shape[0]
            dynamic_positions[:, 1, [0, 1]] = dynamic_positions[:, 1, [1, 0]]
            dynamic_positions = np.flipud(dynamic_positions)
            self.metadata['analysis']['bin_based']['dynamic_positions'] = dynamic_positions
        self.metadata['flip_h'] = not(self.metadata['flip_h'])
        
    def load_metadata(self):
        """
        Loads additional metadata about the image.

        This method retrieves the minimum and maximum values of the x and y coordinates 
        and their intervals from the TapeAnalyzer object, casts them to integers and stores them 
        in the metadata attribute.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        The original metadata of the TapeAnalyzer object is updated with the minimum and maximum x and y values 
        and their intervals, and these values are cast to integers.
        """
        self.metadata["xmin"] = int(self.xmin)
        self.metadata["xmax"] = int(self.xmax)
        self.metadata["x_interval"] = int(self.x_interval)
        self.metadata["ymin"] = int(self.ymin)
        self.metadata["ymax"] = int(self.ymax)
        self.metadata["y_interval"] = int(self.ymax-self.ymin)
        return 

    @classmethod
    def from_dict(cls, image: np.ndarray, metadata: dict):
        """
        Class method to create an instance of the TapeAnalyzer class from provided image data and metadata.

        Parameters
        ----------
        image : np.ndarray
            The image data to initialize the TapeAnalyzer instance. 
        metadata : dict
            Dictionary containing metadata for the TapeAnalyzer instance. 

        Raises
        ------
        Exception
            If the provided image is empty or None.

        Returns
        -------
        TapeAnalyzer
            An instance of TapeAnalyzer initialized with the provided image data and metadata.
            
        Notes
        -----
        This class method provides an alternative way to create an instance of the TapeAnalyzer class, particularly 
        when the necessary image data and metadata are available in advance.
        """
        if image is None:
            raise Exception(
                "The provided image was empty. Maybe change the query criteria")
        cls = cls()
        cls.image = image
        cls.metadata.update(metadata)
        return cls

    def get_image_tilt(self, plot: bool=False) -> float:
        """
        Calculate the tilt angle of the image.

        This function calculates the tilt angle of the image by applying a linear fit to the upper and lower boundaries of the image. 
        It first divides the x-axis into 'n_divisions' segments, then it finds the top and bottom boundaries by searching for points 
        within each segment that are within the top and bottom y-intervals respectively. For each segment, a linear fit is applied to 
        the found boundary points, resulting in a set of slopes. The process is done separately for the top and bottom boundaries. 

        For each set of slopes, the one with the smallest standard deviation of the y-coordinates is selected. 
        If no points are found in a segment, no slope is added for that segment. 

        The final tilt angle is the arctan of the average of the selected top and bottom slopes, converted to degrees.

        Parameters
        ----------
        plot : bool, optional
            If True, the function will plot the boundary conditions that were used for the fit. 
            The top boundary conditions are plotted in blue, and the bottom ones in red. Default is False.

        Returns
        -------
        float
            The tilt angle of the image in degrees.

        Notes
        -----
        If the standard deviation of the y-coordinates of the boundaries used for the fit exceeds 10, 
        the respective y-coordinate is set to the corresponding image boundary (y_max for the top, y_min for the bottom).
        The function also updates the 'crop_y_top', 'crop_y_bottom' and 'image_tilt' keys in the metadata of the TapeAnalyzer instance.
        """
        figsize=(16, 9)
        stds = np.ones((self.metadata.n_divisions-2, 2))*1000
        conditions_top = []
        # conditions_top.append([])
        conditions_bottom = []
        # conditions_bottom.append([])
        
        boundary = self.boundary
        y_min = self.ymin
        y_max = self.ymax

        x_min = self.xmin
        x_max = self.xmax
        m_top = []
        # m_top.append(None)
        m_bottom = []
        # m_bottom.append(None)
        if plot:
            _ = plt.figure(figsize=figsize)
            ax = plt.subplot(111)
            self.plot_boundary(color='black', ax=ax)
            ax.invert_yaxis()
        for idivision in range(self.metadata.n_divisions-2):
            
            y_interval = y_max-y_min
            cond1 = boundary[:, 1] > y_max-y_interval/self.metadata.n_divisions
            cond2 = boundary[:, 1] < y_max+y_interval/self.metadata.n_divisions

            x_interval = x_max-x_min
            cond3 = boundary[:, 0] >= x_min+x_interval/self.metadata.n_divisions*(idivision+1)
            cond4 = boundary[:, 0] <= x_min + \
                x_interval/self.metadata.n_divisions*(idivision+2)

            cond_12 = np.bitwise_and(cond1, cond2)
            cond_34 = np.bitwise_and(cond3, cond4)
            cond_and_top = np.bitwise_and(cond_12, cond_34)

            # This part is to rotate the images

            if sum(cond_and_top) == 0:
                conditions_top.append([])
                m_top.append(None)
                continue
            if plot:
                ax.plot(boundary[cond_and_top][:, 0],
                         boundary[cond_and_top][:, 1], linewidth=3)
            m_top.append(np.polyfit(
                boundary[cond_and_top][:, 0], boundary[cond_and_top][:, 1], 1)[0])
            std_top = np.std(boundary[cond_and_top][:, 1])
            stds[idivision, 0] = std_top
            conditions_top.append(cond_and_top)

        for idivision in range(self.metadata.n_divisions-2):

            cond1 = boundary[:, 1] > y_min-y_interval/self.metadata.n_divisions
            cond2 = boundary[:, 1] < y_min+y_interval/self.metadata.n_divisions

            x_interval = x_max-x_min
            cond3 = boundary[:, 0] >= x_min+x_interval/self.metadata.n_divisions*(idivision+1)
            cond4 = boundary[:, 0] <= x_min + \
                x_interval/self.metadata.n_divisions*(idivision+2)

            cond_12 = np.bitwise_and(cond1, cond2)
            cond_34 = np.bitwise_and(cond3, cond4)
            cond_and_bottom = np.bitwise_and(cond_12, cond_34)
            if sum(cond_and_bottom) == 0:
                conditions_bottom.append([])
                m_bottom.append(None)
                continue
            if plot:
                ax.plot(
                    boundary[cond_and_bottom][:, 0], boundary[cond_and_bottom][:, 1], linewidth=3)
                ax.set_xlim(0, self.image.shape[1])

            m_bottom.append(np.polyfit(
                boundary[cond_and_bottom][:, 0], boundary[cond_and_bottom][:, 1], 1)[0])

            std_bottom = np.std(boundary[cond_and_bottom][:, 1])

            stds[idivision, 1] = std_bottom
            conditions_bottom.append(cond_and_bottom)
        arg_mins = np.argmin(stds, axis=0)
        m_top[arg_mins[0]] = m_bottom[arg_mins[1]] if m_top[arg_mins[0]] is None else m_top[arg_mins[0]]
        m_bottom[arg_mins[1]] = m_top[arg_mins[0]] if m_bottom[arg_mins[1]] is None else m_bottom[arg_mins[1]]
        m = np.average([m_top[arg_mins[0]], m_bottom[arg_mins[1]]])
        cond_and_top = conditions_top[arg_mins[0]]
        cond_and_bottom = conditions_bottom[arg_mins[1]]
        if plot:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.plot(boundary[:, 0], boundary[:, 1], color='black')
            ax.scatter(boundary[cond_and_top][:, 0],
                        boundary[cond_and_top][:, 1], color='blue')
            ax.scatter(boundary[cond_and_bottom][:, 0],
                        boundary[cond_and_bottom][:, 1], color='red')
            ax.set_xlim(0, self.image.shape[1])
            ax.invert_yaxis()
            
        top = boundary[cond_and_top][:, 1]
        bottom = boundary[cond_and_bottom][:, 1]
        self.metadata.crop_y_top = np.min(top) if len(top) != 0 else self.ymax
        self.metadata.crop_y_bottom = np.max(bottom) if len(bottom) !=0 else self.ymin
        if np.min(stds, axis=0)[0] > 10:
            self.metadata.crop_y_top = y_max
        if np.min(stds, axis=0)[1] > 10:
            self.metadata.crop_y_bottom = y_min
            
        angle = np.arctan(m)
        angle_d = np.rad2deg(angle)
        self.metadata.image_tilt = angle_d
        return angle_d

    @property
    def xmin(self) -> int:
        """
        Calculate the minimum x-coordinate value among the boundary points.

        This function computes the minimum value along the x-axis (horizontal direction in image) from the set of 
        coordinates that form the boundary of the image. These boundaries have been identified 
        and stored in the 'boundary' attribute of the object, which is a numpy array of shape (N, 2), 
        where N is the number of boundary points and the 2 columns represent the x and y coordinates, respectively. 

        Returns
        -------
        int
            The minimum x-coordinate value of the boundary points in the image.

        Notes
        -----
        The boundary points should be pre-calculated and stored in the 'boundary' attribute before 
        calling this function. If the 'boundary' attribute is not set, or if it is empty, 
        this function may raise an error or return an unexpected result.
        """
        xmin = self.boundary[:, 0].min()
        return xmin

    @property
    def xmax(self) -> int:
        """
        X coordinate of minimum pixel of the boundary

        Returns
        -------
        xmax : int
            X coordinate of minimum pixel of the boundary

        """
        xmax = self.boundary[:, 0].max()
        return xmax

    @property
    def x_interval(self) -> int:
        """
        Calculate the maximum x-coordinate value among the boundary points.

        This function computes the maximum value along the x-axis (horizontal direction in image) 
        from the set of coordinates that form the boundary of the image. These boundaries have been 
        identified and stored in the 'boundary' attribute of the object, which is a numpy array of 
        shape (N, 2), where N is the number of boundary points and the 2 columns represent the x 
        and y coordinates, respectively. 

        Returns
        -------
        int
            The maximum x-coordinate value of the boundary points in the image.

        Notes
        -----
        The boundary points should be pre-calculated and stored in the 'boundary' attribute before 
        calling this function. If the 'boundary' attribute is not set, or if it is empty, 
        this function may raise an error or return an unexpected result.
        """
        return self.xmax - self.xmin

    @property
    def ymin(self) -> int:
        """
        Calculate the minimum y-coordinate value among the boundary points.

        This function computes the minimum value along the y-axis (vertical direction in image)
        from the set of coordinates that form the boundary of the image. These boundaries have been
        identified and stored in the 'boundary' attribute of the object, which is a numpy array of
        shape (N, 2), where N is the number of boundary points and the 2 columns represent the x
        and y coordinates, respectively.

        Returns
        -------
        int
            The minimum y-coordinate value of the boundary points in the image.

        Notes
        -----
        The boundary points should be pre-calculated and stored in the 'boundary' attribute before
        calling this function. If the 'boundary' attribute is not set, or if it is empty,
        this function may raise an error or return an unexpected result.
        """
        ymin = self.boundary[:, 1].min()
        return ymin

    @property
    def ymax(self) -> int:
        """
        Calculate the maximum y-coordinate value among the boundary points.

        This function computes the maximum value along the y-axis (vertical direction in image)
        from the set of coordinates that form the boundary of the image. These boundaries have been
        identified and stored in the 'boundary' attribute of the object, which is a numpy array of
        shape (N, 2), where N is the number of boundary points and the 2 columns represent the x
        and y coordinates, respectively.

        Returns
        -------
        int
            The maximum y-coordinate value of the boundary points in the image.

        Notes
        -----
        The boundary points should be pre-calculated and stored in the 'boundary' attribute before
        calling this function. If the 'boundary' attribute is not set, or if it is empty,
        this function may raise an error or return an unexpected result.
        """
        ymax = self.boundary[:, 1].max()
        return ymax

    def auto_crop_y(self):
        """
        Automatically crop the image in the y-direction.

        This function removes pixels from the top and bottom of the image based on the values stored 
        in `metadata.crop_y_bottom` and `metadata.crop_y_top`, respectively. This can be useful 
        for focusing on specific regions of interest in the image and removing unnecessary or 
        distracting parts of the image.
        
        Notes
        -----
        The cropping limits are determined by the `metadata.crop_y_bottom` and `metadata.crop_y_top`
        attributes. Before calling this method, these attributes should be calculated or set. If these 
        attributes are not set, the function may throw an error or return an unexpected result.
        
        After cropping, the original image stored in the 'image' attribute is replaced by the cropped
        image. If you need to keep the original image as well, you should create a copy before calling 
        this function.
        
        Returns
        -------
        None
        """
        self.image = self.image[int(
            self.metadata.crop_y_bottom):int(self.metadata.crop_y_top), :]
        return

    def _get_points_from_boundary(self,
                                  x_0: float,
                                  x_1: float,
                                  y_0: float,
                                  y_1: float) -> np.ndarray:
        """
        A private method to extract the points from the object boundary that are 
        within a specific rectangle defined by the input coordinates (x_0, x_1, y_0, y_1).

        Parameters
        ----------
        x_0 : float
            The minimum x-coordinate of the rectangular region.
        x_1 : float
            The maximum x-coordinate of the rectangular region.
        y_0 : float
            The minimum y-coordinate of the rectangular region.
        y_1 : float
            The maximum y-coordinate of the rectangular region.

        Returns
        -------
        np.ndarray
            An array of coordinates within the boundary that lie within the defined 
            rectangular region. Each element of the array is an array of two elements, 
            the x and y coordinates of a point.

        Note
        ----
        The method assumes that the coordinates are stored in a 2D numpy array 
        with each row being a point and the two columns representing the x and y 
        coordinates. The boundary points are filtered based on whether they fall 
        within the rectangular region defined by (x_0, x_1) for the x-coordinate 
        and (y_0, y_1) for the y-coordinate. The resulting array of coordinates 
        is then returned.
        """
        coordinates = self['boundary']
        cond_1 = coordinates[:, 0] >= x_0
        cond_2 = coordinates[:, 0] <= x_1
        cond_x = np.bitwise_and(cond_1, cond_2)
        cond_1 = coordinates[:, 1] >= y_0
        cond_2 = coordinates[:, 1] <= y_1
        cond_y = np.bitwise_and(cond_1, cond_2)
        cond = np.bitwise_and(cond_x, cond_y)
        return coordinates[cond]

    def get_coordinate_based(self,
                             n_points: int=64,
                             x_trim_param: int=6) -> dict:
        """
        This method returns the coordinate-based representation of the detected edge in the image.
        The edge is segmented into 'n_points' horizontal slices, and for each slice, the average
        x and y coordinates of the edge points within that slice are calculated. The method can also
        account for vertically flipped images.

        Parameters
        ----------
        n_points : int, optional
            The number of slices into which the edge is divided. The default is 64.
        x_trim_param : int, optional
            A parameter to determine the range in the x-direction for edge point consideration.
            The x-range is divided by this parameter to define the x-range for edge detection.
            A smaller value of x_trim_param results in a larger x-range. The default is 6.

        Returns
        -------
        dict
            A dictionary containing the calculated coordinate-based representation. The dictionary 
            includes the number of points, the x_trim_param used, and three numpy arrays: 'coordinates' 
            with the average (x, y) coordinates for each slice, 'stds' with the standard deviations 
            of the x-coordinates within each slice, and 'slopes' with the slope and intercept of 
            the least-square linear fit to the edge points within each slice.
        """
        was_flipped = self.metadata['flip_v']
        if was_flipped:
            self.flip_v()
        x_min = self.xmin
        x_max = self.xmax
        x_interval = x_max-x_min
        boundary = self.boundary

        # cond_12 = np.bitwise_and(cond1, cond2)
        x_0 = x_min
        x_1 = x_min+x_interval/x_trim_param*0.95
        cond1 = boundary[:, 0] >= x_0
        cond2 = boundary[:, 0] <= x_1
        cond_12 = np.bitwise_and(cond1, cond2)
        # cond_12 = cond1
        data = np.zeros((n_points, 2))
        stds = np.zeros((n_points,))
        slopes = np.zeros((n_points, 2))
        y_min = self.ymin
        y_max = self.ymax
        y_interval = y_max - y_min
        edge = boundary[cond_12]
        self.metadata['edge_x_std'] = float(np.std(edge[:, 0]))
        self.metadata['edge_x_range'] = int(edge[:, 0].max()-edge[:, 0].min())
        self.debug['cb_points'] = []
        for i_point in range(0, n_points):
            y_start = y_min+i_point*(y_interval/n_points)
            y_end = y_min+(i_point+1)*(y_interval/n_points)
            points = self._get_points_from_boundary(x_0, x_1, y_start, y_end)
            
            if len(points) == 0 and x_trim_param != 1:
                if was_flipped:
                    self.flip_v()
                return self.get_coordinate_based(n_points=n_points, 
                                                 x_trim_param=x_trim_param-1)
            self.debug['cb_points'].append(points)
            data[i_point, :2] = np.average(points, axis=0)
            stds[i_point] = np.std(points[:, 0])
            slopes[i_point, :] = np.polyfit(
                x=points[:, 0], 
                y=points[:, 1],
                deg=1,
                full=True)[0]
        self.metadata['analysis']['coordinate_based'] = {
            "n_points": n_points,
            "x_trim_param": x_trim_param,
            'coordinates': data,
            'stds': stds,
            'slopes': slopes}
        if was_flipped:
            self.flip_v()
        return self['coordinate_based']

    def get_bin_based(self,
                      window_background: int=50,
                      window_tape: int=1000,
                      dynamic_window: bool=True,
                      n_bins: int=10,
                      overlap: int=0,
                      border: str = 'avg',
                      ) -> List[Tuple[int, int]]:
        """
        This method returns the edges detected in the image segmented into a given number of bins. Each bin is a slice 
        of the image, vertically defined and extending horizontally to include a given number of pixels on both sides 
        of the edge. The slices can have overlaps and their horizontal positions can be adjusted dynamically based on 
        the edge location within the slice. 

        Parameters
        ----------
        window_background : int, optional
            Number of pixels to be included in each slice on the background side of the edge. The default is 50.
        window_tape : int, optional
            Number of pixels to be included in each slice on the tape side of the edge. The default is 1000.
        dynamic_window : bool, optional
            Whether to adjust the horizontal position of the slices based on the edge location within the slice. 
            The default is True.
        n_bins : int, optional
            Number of slices into which the edge is divided. The default is 10.
        overlap : int, optional
            The number of rows of overlap between consecutive slices. If positive, slices will overlap, if negative,
            there will be gaps between them. The default is 0.
        border : str, optional
            Determines the method of edge detection within each slice. Options are 'avg' for the average position 
            or 'min' for the minimum position of edge pixels in the slice. The default is 'avg'.

        Returns
        -------
        List[Tuple[int, int]]
            A list of tuples specifying the x-range and y-range of each slice in the format:
            [(x_start, x_end), (y_start, y_end)]
        """
        was_flipped = self.metadata['flip_v']
        if was_flipped:
            self.flip_v()
        overlap = abs(overlap)
        boundary = self.boundary
        x_min = self.xmin
        x_max = self.xmax

        y_min = self.ymin
        y_max = self.ymax

        x_start = x_min-window_background
        x_end = min(x_min+window_tape, self.image.shape[1])

        seg_len = (y_max-y_min)//(n_bins)
        seg_len = (y_max-y_min)/(n_bins)

        segments = []
        dynamic_positions = []
        y_end = y_min
        for iseg in range(0, n_bins):
            y_start = y_end
            y_end = math.ceil(y_min+(iseg+1)*seg_len)
            if dynamic_window:
                cond1 = boundary[:, 1] >= y_start - overlap
                cond2 = boundary[:, 1] <= y_end + overlap
                cond_and = np.bitwise_and(cond1, cond2)
                x_slice = boundary[cond_and, 0]
                if border == 'min' or iseg in [0, n_bins-1]:
                    loc = x_slice.min()
                else:
                    cond_x = x_slice.min() + 500
                    loc = np.average(x_slice[x_slice < cond_x])
                x_start =  loc - window_background
                x_end = loc + window_tape
                if self.image.shape[1] < x_end:
                    diff = self.image.shape[1] - x_end
                    x_start += diff
                    x_end = self.image.shape[1]
            
            ys = y_start - overlap
            ye = y_end + overlap
            
            dynamic_positions.append(
                [[int(x_start), int(x_end)], [int(ys), int(ye)]])

        metadata = {"dynamic_positions": dynamic_positions,
                    "n_bins": n_bins,
                    "overlap": overlap,
                    "dynamic_window": dynamic_window,
                    "window_background": window_background,
                    "window_tape": window_tape}

        self.metadata['analysis']['bin_based'] = metadata
        if was_flipped:
            self.flip_v()
        return dynamic_positions

    def get_max_contrast(self,
                         window_background: int=100,
                         window_tape: int=600) -> np.ndarray:
        """
        This method generates a binary image representation of the detected edge in the image.
        It generates a new image where all pixels are set to 0 (black), except for the ones
        at the detected edge location, which are set to 1 (white).
        
        This can help in visualizing the edge or in performing further analysis, as the edge
        is distinctly highlighted against a uniform background.

        Parameters
        ----------
        window_background : int, optional
            Number of pixels to be included in each slice on the background side of the edge.
            These pixels will be set to black in the resulting image. The default is 100.
        window_tape : int, optional
            Number of pixels to be included in each slice on the tape side of the edge.
            These pixels will be set to black in the resulting image. The default is 600.

        Returns
        -------
        edge_bw : np.ndarray
            A 2D numpy array representing the image. All pixels are black (0), except the ones
            at the detected edge location, which are white (1).
        """
        self.metadata['analysis']['max_contrast'] = {
            "window_background": window_background,
            "window_tape": window_tape}
        return self['max_contrast']

    def __getattr__(self, name):
        return self[name]

    def __getitem__(self, x):
        """
        This method is a special Python method that allows custom classes to 
        implement the square bracket access notation. This method is responsible 
        for returning different attributes of the TapeAnalyzer object, given the 
        attribute name.

        Parameters
        ----------
        x : str
            The name of the attribute. The value of `x` can be one of the 
            following: 'image', 'original_image', 'masked', 'gray_scale', 
            'binarized', 'largest_contour', 'boundary', 'coordinate_based', 
            'edge_bw', 'max_contrast', 'bin_based'.

        Returns
        -------
        Various data types
            The returned data type depends on the attribute name. It can be a 
            numpy array (2D or 3D), a dictionary, a list of numpy arrays or lists, 
            or a single numpy array.

        Raises
        ------
        ValueError
            If the attribute name is not recognized, a ValueError will be raised.
        """
        if x == 'image':
            return self.image
        elif x == 'original_image':
            return self.original_image
        elif x == 'masked':
            return image_tools.remove_background(self.image, 
                                                 self.largest_contour)
        elif x == 'gray_scale':
            return image_tools.to_gray(self.image)
        elif x == 'binarized':
            return image_tools.binerized_mask(
                self['image'], self['masked'])
        elif x == 'largest_contour':
            return self.boundary.reshape(-1, 1, 2)
        elif x == 'boundary':
            return np.asarray(self.metadata['boundary'])
        elif x == 'coordinate_based':
            return self.metadata[
                'analysis']['coordinate_based']
        elif x == 'edge_bw':
            return  cv2.drawContours(
                np.zeros_like(self.image), 
                [self.boundary], 
                0, (255,)*(len(self.image.shape)) , 2)
        elif x == 'max_contrast':
            ret = self['edge_bw']
            x_min = self.xmin
            window_background = self.metadata[
                'analysis']['max_contrast']['window_background']
            window_tape = self.metadata[
                'analysis']['max_contrast']['window_tape']
            x_start = x_min-window_background
            x_end = x_min+window_tape
            pad_x_start = -1*(min(x_start, 0))
            pad_x_end = -1*(min(ret.shape[1]-x_end, 0))
            x_start = max(0, x_start)
            x_end = min(ret.shape[1], x_end)
            ret = ret[:, x_start:x_end]
            ret = np.pad(
                ret,
                ((0, 0),)
                + ((pad_x_start, pad_x_end),) 
                + ((0, 0),)*(len(ret.shape)-2),
                'constant', 
                constant_values=(0,)
                )
            
            return ret
        elif 'bin_based' in x:
            if 'coordinate_based' in x:
                ret = []
                self.debug['cbbb_points'] = []
                n_bins = self.metadata.analysis['bin_based']['n_bins']
                n_points = self.metadata.analysis[
                    'coordinate_based']['n_points']//n_bins
                if self.metadata.analysis['bin_based']['overlap']>50:
                    n_points+=2
                for x, y in self.metadata.analysis[
                    'bin_based']['dynamic_positions']:
                    data = np.zeros((n_points, 2))
                    stds = np.zeros((n_points,))
                    slopes = np.zeros((n_points, 2))
                    y_min = y[0]
                    y_max = y[1]
                    pad_y_min = -1*(min(y_min, 0))
                    pad_y_max = -1*(min(self.image.shape[0]-y_max, 0))
                    y_min = max(y_min, 0)
                    y_max = min(self.image.shape[0], y_max)
                    if self.metadata.padding == 'tape':
                        y_min -= pad_y_max
                        y_max += pad_y_min
                    y_interval = y_max - y_min
                   
                    if n_points > len(self._get_points_from_boundary(
                        x[0], x[1],
                        y_min, y_max
                    )):
                        print('The number of points per bin is smaller than'
                              ' the number of points in this bin.'
                              'Consider decreasing the numbers of points for '
                              'the coordinate based method (n_points).')
                    for i_point in range(n_points):
                        y_start = y_min+i_point*(y_interval/n_points)
                        y_end = y_min+(i_point+1)*(y_interval/n_points)
                        points = []
                        ratios = np.linspace(1, 2, 20)
                        j = 0
                        while len(points) == 0:
                            assert j < len(ratios), ("No points in this window, " 
                                                    "increase the tape window"
                                                    f" {self.metadata.filename}-{self.metadata.split_v['side']}")
                            points = self._get_points_from_boundary(
                                x[0], 
                                x[1]*ratios[j],
                                y_start, 
                                y_end)
                            j += 1
                        
                        
                        self.debug['cbbb_points'].append(points)
                        data[i_point, :2] = np.average(points, axis=0)
                        stds[i_point] = np.std(points[:, 0])
                        slopes[i_point, :] = np.polyfit(
                            x=points[:, 0], 
                            y=points[:, 1], 
                            deg=1, 
                            full=True)[0]
                    ret.append({
                        'coordinates': data,
                        'stds': stds,
                        'slopes': slopes})
                if self.metadata.analysis['bin_based']['overlap']>50:
                    for i, item in enumerate(ret):
                        for key in item:
                            ret[i][key] = np.delete(item[key], [0, -1], 0)
                return ret
            elif x == 'bin_based':
                image = self['image']
            elif 'max_contrast' in x:
                image = self['edge_bw']
            ret = []
            bin_based =  self.metadata['analysis']['bin_based']
            dynamic_positions = np.array(bin_based['dynamic_positions'])
            if bin_based['n_bins'] > 3:
                delta_y = int(np.diff(dynamic_positions[:, 1][1:-2]).mean())
            else:
                delta_y = int(np.diff(dynamic_positions[:, 1]).mean())
            for seg in dynamic_positions:
                x_start, x_end = seg[0]
                y_start, _ = seg[1]
                y_end = y_start + delta_y
                pad_y_start = -1*(min(y_start, 0))
                y_start = max(y_start, 0)
                pad_y_end = -1*(min(self.image.shape[0]-y_end, 0))
                y_end = min(self.image.shape[0], y_end)
                pad_x_start = -1*(min(x_start, 0))
                x_start = max(x_start, 0)
                if self.metadata.padding == 'black':
                    img = image[y_start:y_end, x_start:x_end]
                    # the (len(img.shape)-1) is to adjust to gray scale and rgb
                    img = np.pad(
                        img,
                        ((pad_y_start, pad_y_end),) 
                        + ((pad_x_start, 0),)
                        + ((0, 0),)*(len(img.shape)-2),
                        'constant', 
                        constant_values=(0, )
                        )
                elif self.metadata.padding == 'tape':
                    y_start -= pad_y_end
                    y_end += pad_y_start
                    img = image[y_start:y_end, x_start:x_end]
                ret.append(img)
            n = [x.shape for x in ret]
            if len(set(n)) > 1:
                print(self.metadata.path)
                print(set(n))
            return np.asarray(ret)
        else:
            raise ValueError(f"TapeAnalyzer does not have attrib {x}")    
        

