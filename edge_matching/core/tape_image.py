# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:11:02 2020

@author: ptava
"""

import os
import cv2
import numpy as np


class TapeImage():
    def __init__(self,
                 fname=None,
                 tape_label=None,
                 mask_threshold=60,
                 ):
        self.fname = fname
        self.tape_label = tape_label
        
        self.image = cv2.imread(fname,0)
        self.image_original = self.image.copy()
        self.mask_threshold = mask_threshold
        self.masked = None
            
        
    
    def _get_masked(self):
        """
        
        Populates the masked image with the gray scale threshold
        Returns
        -------
        None.

        """
        self.masked = cv2.inRange(self.gray_scale,self.mask_threshold,255)
        return
    

    @property
    def gray_scale(self):
        """
        

        Returns
        -------
        gray_scale : cv2 object
            Gray Scale image of the input image.

        """
        gray_scale = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        return gray_scale
    
    @property
    def width(self):
        """
        

        Returns
        -------
        int
            Width of the image.

        """
        return self.image.shape[1]
    
    @property
    def height(self):
        """
        

        Returns
        -------
        int
            Height of the image.

        """
        return self.image.shape[0]
    
    
    
    def gaussian_blur(self,window=(15,15)):
        """
        

        Parameters
        ----------
        window : tuple int, optional
            The window in which the gaussian blur is going to be applied.
            The default is (15,15).

        Returns
        -------
        None.

        """
        self.image = cv2.GaussianBlur(self.image,window,0)
        return
    
    def split_vertical(self,pixel_index=None,pick_side='L'):
        """
        

        Parameters
        ----------
        pixel_index : int, optional
            The pixel number at which the image is going to be split. The default is None.
        pick_side : str, optional
            The side in which will over write the image in the class. The default is 'L'.

        Returns
        -------
        None.

        """        
        if pixel_index is None:
            pixel_index = self.width//2
        if pick_side == 'L':
            self.image = self.image[:, 0:pixel_index]
        else : 
            self.image = cv2.flip(self.image[:,pixel_index:self.width],1)
        return
    
    @property
    def contours(self):
        """
        

        Returns
        -------
        contours : list 
            A list of pixels that create the contours in the image

        """
        
        contours,_ = cv2.findContours(self.masked,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        return contours

    @property
    def largest_contour(self):
        """
        

        Returns
        -------
        contour_max_area : list
            A list of pixels forming the contour with the largest area

        """
        
        max_con = 0
        max_con_arg = 0
        for ic in range(len(self.contours)):
            if cv2.contourArea(self.contours[ic]) > max_con :
                max_con = cv2.contourArea(self.contours[ic])
                max_con_arg = ic
        contour_max_area = self.contours[max_con_arg]
        return contour_max_area


    @property
    def boundary(self):
        """
        

        Returns
        -------
        list
            2d array with the list of pixels of the largest contour

        """
        contour_max_area=self.contour_max_area
        return contour_max_area.reshape(contour_max_area.shape[0],2)


    @property
    def edge(self):
        """
        

        Returns
        -------
        edge_bw : 2d array int
            List of pixels that create the boundary.

        """
        zeros = np.zeros_like(self.image)        
        edge_bw = cv2.drawContours(zeros,self.contour_max_area,(255,255,255),2)
        return edge_bw
        
    @property
    def xmin(self):
        """
        Returns
        -------
        xmin : int
            X coordinate of minimum pixel of the boundary

        """
        xmin = self.boundary[:,0].min()
        return xmin
    
    @property
    def xmax(self):
        """
        

        Returns
        -------
        xmax : int
            X coordinate of minimum pixel of the boundary

        """
        xmax = self.boundary[:,0].max()
        return xmax
    
    @property
    def x_interval(self):
        """
        

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
        

        Returns
        -------
        ymin : int
             Y coordinate of minimum pixel of the boundary

        """
        cond1 = self.boundary[:,0]>= self.xmin+self.x_interal/self.n_xsections*1
        cond2 = self.boundary[:,0]<= self.xmin+self.x_interal/self.n_xsections*2
        cond_and = np.bitwise_and(cond1,cond2)
        ymin = int(self.boundary[cond_and,1].min()) # using int because pixel numbers are integers
        return ymin
    
    @property
    def ymax(self):
        """
        

        Returns
        -------
        ymax : int
             Y coordinate of maximum pixel of the boundary

        """
        cond1 = self.boundary[:,0]>= self.xmin+self.x_interal/self.n_xsections*1
        cond2 = self.boundary[:,0]<= self.xmin+self.x_interal/self.n_xsections*2
        cond_and = np.bitwise_and(cond1,cond2)
        ymax = int(self.boundary[cond_and,1].max()) # using int because pixel numbers are integers
        return ymax

    def plot(self,savefig=None):
        """
        
        Plots the image
        Parameters
        ----------
        savefig : str, optional
            path to the file one wants to save the image. The default is None.

        Returns
        -------
        None.

        """
        cv2.imshow(self.tape_label,self.image)
        cv2.waitKey(0)
        if savefig is not None:
            cv2.imwrite(savefig,self.image)

    