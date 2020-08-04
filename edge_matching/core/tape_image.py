# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:11:02 2020

@author: Pedram Tavadze
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
ndivision = 6

class TapeImage():
    def __init__(self,
                 fname=None,
                 tape_label=None,
                 mask_threshold=60,
                 rescale=None,
                 split=True,
                 gaussian_blur=None,
                 split_side='L',
                 split_position=None,
                 ):
        self.fname = fname
        self.tape_label = tape_label
        
        self.image_tilt = None
        self.crop_y_top = None
        self.crop_y_bottom = None
        
        self.image = cv2.imread(fname,0)
        self.image_original = self.image.copy()
        if split :
            self.split_vertical(split_position,split_side)
        if gaussian_blur is  not None :
            self.gaussian_blur(gaussian_blur)
        if rescale is not None:
            self.resize(
                size=(int(self.image.shape[0]*rescale),
                      int(self.image.shape[1]*rescale)))
        self.mask_threshold = mask_threshold
        self.masked = None
        self.colored = cv2.cvtColor(self.image,cv2.COLOR_GRAY2BGR)
        self._get_masked()
        self._get_image_tilt()
        
        
    
    def _get_masked(self):
        """
        
        Populates the masked image with the gray scale threshold
        Returns
        -------
        None.

        """
        self.masked = cv2.inRange(self.image,
                                  self.mask_threshold,
                                  255)
        
        return
    
    @property 
    def binerized_mask(self):
        return cv2.bitwise_and(self.image,
                               self.image,
                                mask=self.masked)
    

    @property
    def gray_scale(self):
        """
        

        Returns
        -------
        gray_scale : cv2 object
            Gray Scale image of the input image.

        """
        if len(self.image.shapelen) >2:
            gray_scale = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        else : 
            gray_scale = self.image
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
    
    
    @property
    def size(self):
        """
        

        Returns
        -------
        int
            Total number of pixels.

        """
        return self.image.size
    
    @property
    def shape(self):
        """
        

        Returns
        -------
        tuple int
            2d tuple with height and width.

        """
        return self.image.shape
    
    
    
    def _get_image_tilt(self,plot=False):
        stds = np.ones((ndivision-1,2))*1000
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

        for idivision in range(1,ndivision-1):

            y_interval = y_max-y_min
            cond1 = boundary[:,1]> y_max-y_interval/ndivision
            cond2 = boundary[:,1]< y_max+y_interval/ndivision
            
            x_interal = x_max-x_min
            cond3 = boundary[:,0]>= x_min+x_interal/ndivision*idivision
            cond4 = boundary[:,0]<= x_min+x_interal/ndivision*(idivision+1)
            
            cond_12 = np.bitwise_and(cond1,cond2)
            cond_34 = np.bitwise_and(cond3,cond4)
            cond_and_top = np.bitwise_and(cond_12,cond_34)
            
            # This part is to rotate the images

            if sum(cond_and_top) == 0 :
                conditions_top.append([])
                
                continue
            if plot:
                plt.plot(boundary[cond_and_top][:,0],boundary[cond_and_top][:,1])
            m_top,b0_top = np.polyfit(boundary[cond_and_top][:,0],boundary[cond_and_top][:,1],1)
            std_top = np.std(boundary[cond_and_top][:,1])
            stds[idivision,0] = std_top
            conditions_top.append(cond_and_top)
            
        for idivision in range(1,ndivision-1):
            
            cond1 = boundary[:,1]> y_min-y_interval/ndivision
            cond2 = boundary[:,1]< y_min+y_interval/ndivision
            
            x_interal = x_max-x_min
            cond3 = boundary[:,0]>= x_min+x_interal/ndivision*idivision
            cond4 = boundary[:,0]<= x_min+x_interal/ndivision*(idivision+1)
            
            cond_12 = np.bitwise_and(cond1,cond2)
            cond_34 = np.bitwise_and(cond3,cond4)
            cond_and_bottom = np.bitwise_and(cond_12,cond_34)
            
            if sum(cond_and_bottom) == 0 :
                
                conditions_bottom.append([])
                continue
            if plot:
                plt.plot(boundary[cond_and_bottom][:,0],boundary[cond_and_bottom][:,1])
            m_bottom,b0_bottom = np.polyfit(boundary[cond_and_bottom][:,0],boundary[cond_and_bottom][:,1],1)
            
            m = np.average([m_top,m_bottom])
            
            std_bottom = np.std(boundary[cond_and_bottom][:,1])
            # print(std_top,std_bottom)
            
            stds[idivision,1] = std_bottom
            conditions_bottom.append(cond_and_bottom)
            if plot:
                plt.figure()
                plt.plot(boundary[:,0],boundary[:,1])
                plt.scatter(boundary[cond_and_top][:,0],boundary[cond_and_top][:,1])
                plt.scatter(boundary[cond_and_bottom][:,0],boundary[cond_and_bottom][:,1])
        
        arg_mins = np.argmin(stds,axis=0)
        
        cond_and_top = conditions_top[arg_mins[0]]
        cond_and_bottom = conditions_bottom[arg_mins[1]]
        
        self.crop_y_top = np.average(boundary[cond_and_top][:,1])
        self.crop_y_bottom = np.average(boundary[cond_and_bottom][:,1])
        
        if np.min(stds,axis=0)[0] > 10 :
            self.crop_y_top = y_max
        if np.min(stds,axis=0)[1] > 10 :
            self.crop_y_bottom = y_min
        angle = np.arctan(m)
        angle_d = np.rad2deg(angle)
        self.image_tilt = angle_d
        return angle_d
        
    def auto_crop_y(self):

        self.image = self.image[int(self.crop_y_bottom):int(self.crop_y_top),:]
        self.colored =  cv2.cvtColor(self.image,cv2.COLOR_GRAY2BGR)
        self._get_masked()
        
        # self._get_image_tilt()
        
    

    def rotate_image(self, angle):
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
        image_center = tuple(np.array(self.image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        self.image = cv2.warpAffine(self.image, rot_mat, self.image.shape[1::-1], flags=cv2.INTER_LINEAR)
        self.colored =  cv2.cvtColor(self.image,cv2.COLOR_GRAY2BGR)
        self._get_masked()

        return 
    
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
        self.colored =  cv2.cvtColor(self.image,cv2.COLOR_GRAY2BGR)
        return
    
    def split_vertical(self,pixel_index=0.5,pick_side='L'):
        """
        

        Parameters
        ----------
        pixel_index : int, optional
            fraction in which the image is going to be split. The default is 0.5.
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
        self.colored =  cv2.cvtColor(self.image,cv2.COLOR_GRAY2BGR)
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
        contour_max_area=self.largest_contour
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
        n_xsections = 6
        cond1 = self.boundary[:,0]>= self.xmin+self.x_interval/n_xsections*1
        cond2 = self.boundary[:,0]<= self.xmin+self.x_interval/n_xsections*2
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
        n_xsections = 6
        cond1 = self.boundary[:,0]>= self.xmin+self.x_interval/n_xsections*1
        cond2 = self.boundary[:,0]<= self.xmin+self.x_interval/n_xsections*2
        cond_and = np.bitwise_and(cond1,cond2)
        ymax = int(self.boundary[cond_and,1].max()) # using int because pixel numbers are integers
        return ymax
    
    def resize(self,size=None):
        if size is None:
            return 
        else:
            self.image = cv2.resize(self.image,size)

    def show(self,savefig=None,cmap='gray'):
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
        plt.imshow(self.image,cmap=cmap)
        if savefig is not None:
            cv2.imwrite(savefig,self.image)
        return 
    
    def show_binarized(self,savefig=None,cmap='gray'):
        plt.imshow(self.binerized_mask,cmap=cmap)
        return 

    def plot_boundary(self,savefig=None,color='r'):
        """
        

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
        plt.plot(self.boundary[:,0],self.boundary[:,1],c=color)
        return

    def coordinate_based(self,npoints=1000,x_trim_param=2,plot=False):
        """
        

        Parameters
        ----------
        npoints : int, optional
            Number of points to be selected on the edge. The default is 1000.
        x_trim_param : int, optional
            The x direction of the edge will be divided by this number and only the first one is selected. The default is 6.

        Returns
        -------
        None.

        """
        x_min = self.xmin
        x_max = self.xmax
        x_interal = x_max-x_min
        boundary = self.boundary

        cond1 = boundary[:,0]>= x_min+x_interal/x_trim_param*0
        cond2 = boundary[:,0]<= x_min+x_interal/x_trim_param*1
        cond_12 = np.bitwise_and(cond1,cond2)
        # cond_12 = cond1
        data = np.zeros((npoints,2))
        y_min = self.ymin
        y_max = self.ymax
        y_interval = y_max - y_min
        edge = boundary[cond_12]
        

        
        
        for ipoint in range(0,npoints):
            y_start = y_min+ipoint*(y_interval/npoints)
            y_end   = y_min+(ipoint+1)*(y_interval/npoints)
            cond1 = edge[:,1] >= y_start
            cond2 = edge[:,1] <= y_end
            cond_and = np.bitwise_and(cond1,cond2)


            data[ipoint,:] = np.average(edge[cond_and],axis=0)
        if plot:
            plt.figure(figsize=(1.65,10))
            plt.scatter(data[:,0],data[:,1],s=1)
        
        return data
    
    def weft_based(self,
                   window_background=20,
                   window_tape=300,
                   dynamic_window=False,
                   size=(300,30),
                   nsegments=4,
                   plot=False,
                   ):
        """
        

        Parameters
        ----------
        window_background : int, optional
            Number of pixels to be included in each segment in the backgroung side of the image. The default is 20.
        window_tape : int, optional
            Number of pixels to be included in each segment in the tape side of the image. The default is 300.
        dynamic_window : TYPE, optional
            Whether the windows move in each segment. The default is False.
        size : 2d tuple integers, optional
            The size of each segment in pixels. The default is (300,30).
        nsegments : int, optional
            Number of segments that the tape is going to divided into. The default is 4.
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
        
        x_start = x_min-window_background
        x_end = x_min+window_tape

        seg_len = (y_max-y_min)//(nsegments)
        
        segments = []
        plot_segmets = []
        for iseg in range(0,nsegments):
            y_start = y_min+iseg*seg_len
            y_end =   y_min+(iseg+1)*seg_len
            if dynamic_window:
                cond1 = boundary[:,1]>= y_start
                cond2 = boundary[:,1]<= y_end
                cond_and = np.bitwise_and(cond1,cond2)
                x_start =  boundary[cond_and,0].min()
                x_end   =  x_start + window_tape
#            cv2.imwrite('%s_%d.tif'%(fname[:-4],iseg),im_color[y_start:y_end,x_start:x_end])
            isection = self.binerized_mask[y_start:y_end,x_start:x_end]
            # isection = cv2.resize(isection,(size[1],size[0]))
            segments.append(isection)
            
            isection = cv2.copyMakeBorder(isection,1,1,1,1,cv2.BORDER_CONSTANT,value=[0,0,0])
            plot_segmets.append(isection)
            # segments.append(cv2.cvtColor(isection,cv2.COLOR_BGR2GRAY))
            
        if plot:
            plt.figure()
            plt.imshow(cv2.vconcat(plot_segmets),cmap='gray')
        return segments
    
    def max_contrast(self,
                     window_background=20,
                     window_tape=300,
                     plot=False):
        """
        

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
        zeros = np.zeros_like(self.image)        
        edge_bw = cv2.drawContours(zeros,[self.boundary],0,(255,255,255),2)
        x_min = self.xmin
        
        x_start = x_min-window_background
        x_end = x_min+window_tape
        edge_bw = edge_bw[:,x_start:x_end]
        if plot:
            plt.figure()
            plt.imshow(edge_bw,cmap='gray')
        return edge_bw
    