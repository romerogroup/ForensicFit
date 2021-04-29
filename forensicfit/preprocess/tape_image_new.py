# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:11:02 2020

@author: Pedram Tavadze
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from ..utils import image_tools
from ..core import Tape


class TapeImage():
    def __init__(self,
                 fname,
                 tape_label=None,
                 flip=False,
                 mask_threshold=60,
                 rescale=None,
                 split=False,
                 gaussian_blur=(15, 15),
                 split_side='L',
                 split_position=None,
                 ndivision=6,
                 auto_crop=True,
                 verbose=True,
                 ):
        """
        TapeImage is a class created for tape images to be preprocessed for 
        Machine Learning. This Class detects the edges, auto crops the image 
        and returns the results in 3 different method coordinate_based, 
        weft_based and max_contrast. 

        Parameters
        ----------
        fname : str
            fname defines the path to the image file. This can be any format the `opencv <https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html>`_ supports.

        tape_label : str, optional
            Label to this specific image. The default is None.
        flip : bool, optional
            Applies inversion
        mask_threshold : int, optional
            Inorder to find the boundaries of the tape, the algorithm changes every pixel with a value lower than mask_threshold to 0(black). The default is
            60.
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



        """
        self.fname = fname
        self.verbose = verbose
        if not os.path.exists(fname):
            raise Exception("File %s does not exist" % fname)

        self.tape_label = tape_label
        self.flip = flip
        self.image_tilt = None
        self.crop_y_top = None
        self.crop_y_bottom = None
        self.ndivision = ndivision
        if verbose:
            print("openning %s" % self.fname)
        self.image = cv2.imread(fname)

        self.image_original = self.image.copy()
        if self.flip:
            self.image = cv2.flip(self.image, 0)

        if split:
            if verbose:
                print("splitting and selecting %s" % split_side)
            self.image = image_tools.split_vertical(
                self.image, split_position, split_side)
        if gaussian_blur is not None:
            if verbose:
                print("applying Gaussian Blur")
            self.image = image_tools.gaussian_blur(
                self.image, window=gaussian_blur)
        if rescale is not None:
            self.image = image_tools.resize(self.image,
                                            size=(int(self.image.shape[0]*rescale),
                                                  int(self.image.shape[1]*rescale)))
        self.mask_threshold = mask_threshold
        self.masked = None
        # self.colored = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        if verbose:
            print("getting the mask")
        self.masked = image_tools.get_masked(self.image, self.mask_threshold)
        self.binarized_mask = image_tools.get_masked(
            self.image, self.mask_threshold)
        self.gray_scale = image_tools.gray_scale(self.image)
        if verbose:
            print("calculating the tilt")
        self.contours = image_tools.contours(self.image, self.mask_threshold)
        self.largest_contour = image_tools.largest_contour(self.contours)
        self.boundary = self.largest_contour.reshape(
            self.largest_contour.shape[0], 2)
        self.get_image_tilt()
        if auto_crop:
            self.auto_crop_y()
            self.contours = image_tools.contours(self.image, self.mask_threshold)
            self.largest_contour = image_tools.largest_contour(self.contours)
            self.boundary = self.largest_contour.reshape(
                self.largest_contour.shape[0], 2)
        self.binerized_mask = image_tools.binerized_mask(self.image, self.masked)
        
        return

    def get_image_tilt(self, plot=False):
        """
        This function calculates the degree in which the tape is tilted with 
        respect to the horizontal line.

        Parameters
        ----------
        plot : bool, optional
            Plot the segmentation as the image tilt is being calculated. The 
            default is False.

        Returns
        -------
        angle_d : TYPE
            DESCRIPTION.

        """
        stds = np.ones((self.ndivision-1, 2))*1000
        conditions_top = []
        conditions_top.append([])
        conditions_bottom = []
        conditions_bottom.append([])
        boundary = self.boundary
        y_min = self.ymin
        y_max = self.ymax

        x_min = self.xmin
        x_max = self.xmax
        if plot:
            _ = plt.figure()

        for idivision in range(1, self.ndivision-1):

            y_interval = y_max-y_min
            cond1 = boundary[:, 1] > y_max-y_interval/self.ndivision
            cond2 = boundary[:, 1] < y_max+y_interval/self.ndivision

            x_interal = x_max-x_min
            cond3 = boundary[:, 0] >= x_min+x_interal/self.ndivision*idivision
            cond4 = boundary[:, 0] <= x_min + \
                x_interal/self.ndivision*(idivision+1)

            cond_12 = np.bitwise_and(cond1, cond2)
            cond_34 = np.bitwise_and(cond3, cond4)
            cond_and_top = np.bitwise_and(cond_12, cond_34)

            # This part is to rotate the images

            if sum(cond_and_top) == 0:
                conditions_top.append([])

                continue
            if plot:
                plt.plot(boundary[cond_and_top][:, 0],
                         boundary[cond_and_top][:, 1], linewidth=3)
            m_top, b0_top = np.polyfit(
                boundary[cond_and_top][:, 0], boundary[cond_and_top][:, 1], 1)
            std_top = np.std(boundary[cond_and_top][:, 1])
            stds[idivision, 0] = std_top
            conditions_top.append(cond_and_top)

        for idivision in range(1, self.ndivision-1):

            cond1 = boundary[:, 1] > y_min-y_interval/self.ndivision
            cond2 = boundary[:, 1] < y_min+y_interval/self.ndivision

            x_interal = x_max-x_min
            cond3 = boundary[:, 0] >= x_min+x_interal/self.ndivision*idivision
            cond4 = boundary[:, 0] <= x_min + \
                x_interal/self.ndivision*(idivision+1)

            cond_12 = np.bitwise_and(cond1, cond2)
            cond_34 = np.bitwise_and(cond3, cond4)
            cond_and_bottom = np.bitwise_and(cond_12, cond_34)

            if sum(cond_and_bottom) == 0:

                conditions_bottom.append([])
                continue
            if plot:
                plt.plot(
                    boundary[cond_and_bottom][:, 0], boundary[cond_and_bottom][:, 1], linewidth=3)
            m_bottom, b0_bottom = np.polyfit(
                boundary[cond_and_bottom][:, 0], boundary[cond_and_bottom][:, 1], 1)

            m = np.average([m_top, m_bottom])

            std_bottom = np.std(boundary[cond_and_bottom][:, 1])

            stds[idivision, 1] = std_bottom
            conditions_bottom.append(cond_and_bottom)

        arg_mins = np.argmin(stds, axis=0)

        cond_and_top = conditions_top[arg_mins[0]]
        cond_and_bottom = conditions_bottom[arg_mins[1]]
        if plot:
            plt.figure()
            plt.plot(boundary[:, 0], boundary[:, 1], color='black')
            plt.scatter(boundary[cond_and_top][:, 0],
                        boundary[cond_and_top][:, 1], color='blue')
            plt.scatter(boundary[cond_and_bottom][:, 0],
                        boundary[cond_and_bottom][:, 1], color='red')

        self.crop_y_top = np.average(boundary[cond_and_top][:, 1])
        self.crop_y_bottom = np.average(boundary[cond_and_bottom][:, 1])

        if np.min(stds, axis=0)[0] > 10:
            self.crop_y_top = y_max
        if np.min(stds, axis=0)[1] > 10:
            self.crop_y_bottom = y_min
        angle = np.arctan(m)
        angle_d = np.rad2deg(angle)
        self.image_tilt = angle_d

        return angle_d

    def auto_crop_y(self):
        """
        This method automatically crops the image in y direction (top and bottom)

        Returns
        -------
        None.

        """
        self.image = self.image[int(
            self.crop_y_bottom):int(self.crop_y_top), :]
        image_tools.get_masked(self.image, self.mask_threshold)


    @property
    def edge(self):
        """
         List of pixels that create the boundary.

        Returns
        -------
        edge_bw : 2d array int
            List of pixels that create the boundary.

        """
        zeros = np.zeros_like(self.image)
        edge_bw = cv2.drawContours(
            zeros, self.largest_contour, (255, 255, 255), 2)
        return edge_bw

    @property
    def xmin(self):
        """
        X coordinate of minimum pixel of the boundary

        Returns
        -------
        xmin : int
            X coordinate of minimum pixel of the boundary

        """
        xmin = self.boundary[:, 0].min()
        return xmin

    @property
    def xmax(self):
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
    def x_interval(self):
        """
        interval of the coordinates in X direction

        Returns
        -------
        x_interval : int
            interval of the coordinates in X direction

        """
        x_interval = self.xmax-self.xmin
        return x_interval

    @property
    def ymin(self):
        """
        Y coordinate of minimum pixel of the boundary

        Returns
        -------
        ymin : int
             Y coordinate of minimum pixel of the boundary

        """
        n_xsections = 6
        cond1 = self.boundary[:, 0] >= self.xmin+self.x_interval/n_xsections*1
        cond2 = self.boundary[:, 0] <= self.xmin+self.x_interval/n_xsections*2
        cond_and = np.bitwise_and(cond1, cond2)
        # using int because pixel numbers are integers
        ymin = int(self.boundary[cond_and, 1].min())
        return ymin

    @property
    def ymax(self):
        """
        Y coordinate of maximum pixel of the boundary

        Returns
        -------
        ymax : int
             Y coordinate of maximum pixel of the boundary

        """
        n_xsections = 6
        cond1 = self.boundary[:, 0] >= self.xmin+self.x_interval/n_xsections*1
        cond2 = self.boundary[:, 0] <= self.xmin+self.x_interval/n_xsections*2
        cond_and = np.bitwise_and(cond1, cond2)
        # using int because pixel numbers are integers
        ymax = int(self.boundary[cond_and, 1].max())
        return ymax

    def show(self, savefig=None, cmap='gray'):
        """
        Plots the image

        Parameters
        ----------
        savefig : str, optional
            path to the file one wants to save the image. The default is None.
        cmap : str, optional
            The color map in which the image is shown. The default is 'gray'.

        Returns
        -------
        None.

        """
        plt.imshow(self.image, cmap=cmap)
        if savefig is not None:
            cv2.imwrite(savefig, self.image)
        return

    def show_binarized(self, savefig=None, cmap='gray'):
        """
        This function returns the image as the background 

        Parameters
        ----------
        savefig : TYPE, optional
            DESCRIPTION. The default is None.
        cmap : TYPE, optional
            DESCRIPTION. The default is 'gray'.

        Returns
        -------
        None.

        """
        plt.imshow(self.binerized_mask, cmap=cmap)
        return

    def plot_boundary(self, savefig=None, color='r'):
        """
        This function plots the detected boundary of the image. 

        Parameters
        ----------
        savefig : str, optional
            path to save the plot. The default is None.

        Returns
        -------
        None.

        """
        # if savefig is not None:
        #     plt.savefig()
        plt.plot(self.boundary[:, 0], self.boundary[:, 1], c=color)
        return

    def coordinate_based(self, npoints=1000, x_trim_param=2, plot=False):
        """
        This method returns the data of the detected edge as a set of points 
        in 2d plain with (x,y)

        Parameters
        ----------
        npoints int, optional
                Number of points to be selected on the edge. The default is 
                1000.
        x_trim_param int, optional
                The x direction of the edge will be divided by this number and
                only the first one is selected. The default is 6.

        Returns:

            None.

        """
        x_min = self.xmin
        x_max = self.xmax
        x_interal = x_max-x_min
        boundary = self.boundary

        cond1 = boundary[:, 0] >= x_min+x_interal/x_trim_param*0
        cond2 = boundary[:, 0] <= x_min+x_interal/x_trim_param*1
        cond_12 = np.bitwise_and(cond1, cond2)
        # cond_12 = cond1
        data = np.zeros((npoints, 2))
        y_min = self.ymin
        y_max = self.ymax
        y_interval = y_max - y_min
        edge = boundary[cond_12]

        for ipoint in range(0, npoints):
            y_start = y_min+ipoint*(y_interval/npoints)
            y_end = y_min+(ipoint+1)*(y_interval/npoints)
            cond1 = edge[:, 1] >= y_start
            cond2 = edge[:, 1] <= y_end
            cond_and = np.bitwise_and(cond1, cond2)

            data[ipoint, :] = np.average(edge[cond_and], axis=0)
        if plot:
            # plt.figure(figsize=(1.65,10))
            plt.figure()
            self.show()
            plt.scatter(data[:, 0], data[:, 1], s=1, color='red')
            plt.xlim(data[:, 0].min()*0.9, data[:, 0].max()*1.1)

        return data

    def weft_based(self,
                   window_background=50,
                   window_tape=200,
                   dynamic_window=False,
                   size=(300, 30),
                   nsegments=4,
                   plot=False,
                   ):
        """
        This method returns the detected edge as a set of croped images from 
        the edge. The number if images is defined by nsegments. The goal is 
        to try to match the segmentation to the wefts of the tape. 
        In the future this method will try to detecte the wefts automatically.


        Parameters
        ----------
        window_background : int, optional
            Number of pixels to be included in each segment in the backgroung 
            side of the image. The default is 20.
        window_tape : int, optional
            Number of pixels to be included in each segment in the tape side 
            of the image. The default is 300.
        dynamic_window : TYPE, optional
            Whether the windows move in each segment. The default is False.
        size : 2d tuple integers, optional
            The size of each segment in pixels. The default is (300,30).
        nsegments : int, optional
            Number of segments that the tape is going to divided into. The 
            default is 4.
        plot : bool, optional
            Whether or not to plot the divided image. The default is False.


        Returns
        -------
        An array of 2d numpy arrays with images of each segment.

        """
        boundary = self.boundary
        x_min = self.xmin
        x_max = self.xmax

        y_min = self.ymin
        y_max = self.ymax
        y_min = 0
        y_max = self.image.shape[0]

        x_start = x_min-window_background
        x_end = x_min+window_tape

        seg_len = (y_max-y_min)//(nsegments)
        seg_len = (y_max-y_min)/(nsegments)

        segments = []
        plot_segmets = []
        dynamic_positions = []
        y_end = 0
        for iseg in range(0, nsegments):
            # y_start = y_min+iseg*seg_len
            y_start = y_end
            y_end = math.ceil(y_min+(iseg+1)*seg_len)
            if dynamic_window:
                cond1 = boundary[:, 1] >= y_start
                cond2 = boundary[:, 1] <= y_end
                cond_and = np.bitwise_and(cond1, cond2)
                x_start = boundary[cond_and, 0].min()
                x_end = x_start + window_tape
#            cv2.imwrite('%s_%d.tif'%(fname[:-4],iseg),im_color[y_start:y_end,x_start:x_end])
            isection = self.binerized_mask[y_start:y_end, x_start:x_end]
            dynamic_positions.append([[x_start, x_end], [y_start, y_end]])

            isection = cv2.resize(isection, (size[1], size[0]))
            segments.append(isection)

            isection = cv2.copyMakeBorder(
                isection, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            plot_segmets.append(isection)
            # segments.append(cv2.cvtColor(isection,cv2.COLOR_BGR2GRAY))

        if plot:
            plt.figure()
            plt.imshow(cv2.vconcat(plot_segmets), cmap='gray')
            plt.figure()

            self.show(cmap='gray')
            for iseg in dynamic_positions:
                # y = (iseg[1][0]+iseg[1][1])/2
                y1 = iseg[1][0]
                y2 = iseg[1][1]
                x1 = iseg[0][0]
                x2 = iseg[0][1]
                plt.plot([x1, x1], [y1, y2], color='red')
                plt.plot([x2, x2], [y1, y2], color='red')
                plt.plot([x1, x2], [y1, y1], color='red')
                plt.plot([x1, x2], [y2, y2], color='red')

                # plt.plot(iseg[0],[y,y],color='red')
        return segments

    def max_contrast(self,
                     window_background=20,
                     window_tape=300,
                     size=None,
                     plot=False):
        """
        This method returns the detected image as a black image with only the 
        boundary being white.

        Parameters
        ----------
        window_background : int, optional
            Number of pixels to be included in each segment in the backgroung side of the image. The default is 20.
        window_tape : int, optional
            Number of pixels to be included in each segment in the tape side of the image. The default is 300.
        plot : bool, optional
            Whether or not to plot the divided image. The default is False.

        Returns
        -------
        edge_bw : 2d numpy array
            A black image with all the pixels on the edge white.

        """
        if self.verbose:
            print("getting max contrast")
        zeros = np.zeros_like(self.image)
        edge_bw = cv2.drawContours(
            zeros, [self.boundary], 0, (255, 255, 255), 2)
        x_min = self.xmin

        x_start = x_min-window_background
        x_end = x_min+window_tape
        edge_bw = edge_bw[:, x_start:x_end]
        if size is not None:
            edge_bw = cv2.resize(edge_bw, (size[1], size[0]))
        if plot:
            plt.figure()
            plt.imshow(edge_bw, cmap='gray')
        return edge_bw
