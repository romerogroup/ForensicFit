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
        return cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        

    @property
    def masked(self):
        return cv2.inRange(self.gray_scale,self.mask_threshold,255)

    
    @property
    def contours(self):
        contours,_ = cv2.findContours(self.masked,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        return contours
    
    
    
    @property
    def contour_max_area(self):
        max_con = 0
        max_con_arg = 0
        for ic in range(len(self.contours)):
            if cv2.contourArea(self.contours[ic]) > max_con :
                max_con = cv2.contourArea(self.contours[ic])
                max_con_arg = ic
        return self.contours[max_con_arg]

    @property
    def boundary(self):
        contour_max_area=self.contour_max_area
        return contour_max_area.reshape(contour_max_area.shape[0],2)


    @property
    def edge(self):
        zeros = np.zeros_like(self.image)        
        edge_bw = cv2.drawContours(zeros,self.contour_max_area,(255,255,255),2)
        return edge_bw
        
    @property
    def xmin(self):
        return self.boundary[:,0].min()
    
    @property
    def xmax(self):
        return self.boundary[:,0].max()
    
    @property
    def x_interval(self):
        return self.xmax-self.xmin
    
    @property
    def ymin(self):
        cond1 = self.boundary[:,0]>= self.xmin+self.x_interal/self.n_xsections*1
        cond2 = self.boundary[:,0]<= self.xmin+self.x_interal/self.n_xsections*2
        cond_and = np.bitwise_and(cond1,cond2)       
        return int(self.boundary[cond_and,1].min()) # using int because pixel numbers are integers
    
    @property
    def ymax(self):
        cond1 = self.boundary[:,0]>= self.xmin+self.x_interal/self.n_xsections*1
        cond2 = self.boundary[:,0]<= self.xmin+self.x_interal/self.n_xsections*2
        cond_and = np.bitwise_and(cond1,cond2)       
        return int(self.boundary[cond_and,1].max()) # using int because pixel numbers are integers
    

    @property
    def dy(self):
        return (self.ymax-self.ymin)//(self.nsegments)
    

    @property
    def segments(self):
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
    #            cv2.imwrite('%s_%d.tif'%(fname[:-4],iseg),im_color[y_start:y_end,x_start:x_end])
                isection = self.image[y_start:y_end,x_start:x_end]
                isection = cv2.resize(isection,(self.ny_pixel,self.ny_pixel))
    #            isection = cv2.copyMakeBorder(isection,1,1,1,1,cv2.BORDER_CONSTANT,value=[0,0,0])
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
