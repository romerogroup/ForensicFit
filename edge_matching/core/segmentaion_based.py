# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:41:08 2020

@author: Pedram tavadze
"""

__author__ = "Pedram Tavadze"

import cv2 
import numpy as np

class SegmentationBased:
    def __init__(self,
                 image,
                 nsegments,
                 boundary,
                 window_tape,
                 window_bg,
                 dynamic_window=False,
                 mask_threshold=60,
                 n_xsections=6,
                 nx_pixel=None,
                 ny_pixel=None,
                 ):
        
        self.image = image
        self.nsegments = nsegments
        self.boundary = boundary
        self.window_bg = window_bg
        self.window_tape = window_tape
        self.dynamic_window = dynamic_window
        self.mask_threshold = mask_threshold
        self.nx_pixel = nx_pixel 
        self.ny_pixel = ny_pixel # for each segment
        
        
        
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
    def masked(self):
        """
        

        Returns
        -------
        masked : cv2 obj
            Masked imge in th 256 gray scale.

        """
        masked = cv2.inRange(self.gray_scale,self.mask_threshold,255)
        return masked
        
            

    
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
    def contour_max_area(self):
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
        edge_bw : TYPE
            DESCRIPTION.

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
    

    @property
    def dy(self):
        return (self.ymax-self.ymin)//(self.nsegments)
    

    @property
    def segments(self):
        """
        

        Returns
        -------
        segments : list int
            returns a list, where each element is a cv2 object.

        """
        segments= []
        if not self.dynamic_window:
            x_start = self.xmin-self.window_bg
            x_end   = self.xmin+self.window_tape
            edge_bw= self.edge[:,x_start:x_end]
            edge_bw = cv2.cvtColor(edge_bw,cv2.COLOR_BGR2GRAY)
            # edge_bw = cv2.resize(edge_bw,(self.nx_pixel,self.ny_pixel))
            for iseg in range(self.nsegments):
                y_start = self.ymin+iseg*self.dy
                y_end   = y_start+self.dy
                # cv2.imwrite('%s_%d.tif'%(fname[:-4],iseg),im_color[y_start:y_end,x_start:x_end])
                isection = self.image[y_start:y_end,x_start:x_end]
                isection = cv2.resize(isection,(self.ny_pixel,self.ny_pixel))
                # isection = cv2.copyMakeBorder(isection,1,1,1,1,cv2.BORDER_CONSTANT,value=[0,0,0])
                segments.append(cv2.cvtColor(isection,cv2.COLOR_BGR2GRAY))
        return segments

    def plot_all(self,savefig=None):
        segs = []
        for iseg in self.segments:
            isection = cv2.copyMakeBorder(iseg,1,1,1,1,
                                          cv2.BORDER_CONSTANT,
                                          value=[0,0,0])
            segs.append(isection)
        
        to_plot = cv2.vconcat(segs)
        cv2.imshow('Segmented',to_plot)
        cv2.waitKey(0)
        if savefig is not None:
            cv2.imwrite(savefig,to_plot)
