# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:11:02 2020

@author: Pedram Tavadze
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


class Tape():
    def __init__(self,
                 tape_label=None,
                 image=None,
                 max_contrast=None,
                 weft_based=None,
                 coordinate_based=None,
                 meta_data={},
                 ):

        self.fname = fname
        self.verbose = verbose
        
        self.tape_label = tape_label
        self.flip = flip
        self.image_tilt = None
        self.crop_y_top = None
        self.crop_y_bottom = None
        if verbose:
            print("openning %s" % self.fname)
        self.image = cv2.imread(fname,0)
        if self.flip:
            self.image=cv2.flip(self.image, 0)
        self.image_original = self.image.copy()
        if split :
            if verbose:
                print("splitting and selecting %s"%split_side)
            self.split_vertical(split_position,split_side)
        if gaussian_blur is not None :
            if verbose:
                print("applying Gaussian Blur")
            self.gaussian_blur(gaussian_blur)
        if rescale is not None:
            self.resize(
                size=(int(self.image.shape[0]*rescale),
                      int(self.image.shape[1]*rescale)))
        self.mask_threshold = mask_threshold
        self.masked = None
        self.colored = cv2.cvtColor(self.image,cv2.COLOR_GRAY2BGR)
        if verbose:
            print("getting the mask")
        self._get_masked()
        if verbose:
            print("calculating the tilt")
        self.get_image_tilt()
        
        

    
   



    
    @property
    def width(self):
        """
        Width of the image.

        Returns
        -------
        int
            Width of the image.

        """
        return self.image.shape[1]
    
    @property
    def height(self):
        """
        Height of the image.

        Returns
        -------
        int
            Height of the image.

        """
        return self.image.shape[0]
    
    
    @property
    def size(self):
        """
        Total number of pixels.

        Returns
        -------
        int
            Total number of pixels.

        """
        return self.image.size
    
    @property
    def shape(self):
        """
        2d tuple with height and width.

        Returns
        -------
        tuple int
            2d tuple with height and width.

        """
        return self.image.shape
    



    @property
    def boundary(self):
        """
        2d array with the list of pixels of the largest contour

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
         List of pixels that create the boundary.

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
        X coordinate of minimum pixel of the boundary
        
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
        X coordinate of minimum pixel of the boundary

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
        cond1 = self.boundary[:,0]>= self.xmin+self.x_interval/n_xsections*1
        cond2 = self.boundary[:,0]<= self.xmin+self.x_interval/n_xsections*2
        cond_and = np.bitwise_and(cond1,cond2)
        ymin = int(self.boundary[cond_and,1].min()) # using int because pixel numbers are integers
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
        cond1 = self.boundary[:,0]>= self.xmin+self.x_interval/n_xsections*1
        cond2 = self.boundary[:,0]<= self.xmin+self.x_interval/n_xsections*2
        cond_and = np.bitwise_and(cond1,cond2)
        ymax = int(self.boundary[cond_and,1].max()) # using int because pixel numbers are integers
        return ymax
    


    
    # @property
    # def slope(self):
    #     return np.diff(self.boundary[1:-1:10,1])/np.diff(self.boundary[1:-1:10,0])
            

    def show(self,savefig=None,cmap='gray'):
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
        plt.imshow(self.image,cmap=cmap)
        if savefig is not None:
            cv2.imwrite(savefig,self.image)
        return 
    
    def show_binarized(self,savefig=None,cmap='gray'):
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
        plt.imshow(self.binerized_mask,cmap=cmap)
        return 

    def plot_boundary(self,savefig=None,color='r'):
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
        plt.plot(self.boundary[:,0],self.boundary[:,1],c=color)
        return

    def coordinate_based(self,npoints=1000,x_trim_param=2,plot=False):

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
            # plt.figure(figsize=(1.65,10))
            plt.figure()
            self.show()
            plt.scatter(data[:,0],data[:,1],s=1,color='red')
            plt.xlim(data[:,0].min()*0.9,data[:,0].max()*1.1)
            
        
        return data
    
    def weft_based(self,
                   window_background=50,
                   window_tape=200,
                   dynamic_window=False,
                   size=(300,30),
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
        for iseg in range(0,nsegments):
            # y_start = y_min+iseg*seg_len
            y_start = y_end
            y_end =   math.ceil(y_min+(iseg+1)*seg_len)
            if dynamic_window:
                cond1 = boundary[:,1]>= y_start
                cond2 = boundary[:,1]<= y_end
                cond_and = np.bitwise_and(cond1,cond2)
                x_start =  boundary[cond_and,0].min()
                x_end   =  x_start + window_tape
#            cv2.imwrite('%s_%d.tif'%(fname[:-4],iseg),im_color[y_start:y_end,x_start:x_end])
            isection = self.binerized_mask[y_start:y_end,x_start:x_end]
            dynamic_positions.append([[x_start,x_end],[y_start,y_end]])

            isection = cv2.resize(isection,(size[1],size[0]))
            segments.append(isection)
            
            isection = cv2.copyMakeBorder(isection,1,1,1,1,cv2.BORDER_CONSTANT,value=[0,0,0])
            plot_segmets.append(isection)
            # segments.append(cv2.cvtColor(isection,cv2.COLOR_BGR2GRAY))
            
        if plot:
            plt.figure()
            plt.imshow(cv2.vconcat(plot_segmets),cmap='gray')
            plt.figure()

            self.show(cmap='gray')
            for iseg in dynamic_positions:
                # y = (iseg[1][0]+iseg[1][1])/2
                y1 = iseg[1][0]
                y2 = iseg[1][1]
                x1 = iseg[0][0]
                x2 = iseg[0][1]
                plt.plot([x1,x1],[y1,y2],color='red')
                plt.plot([x2,x2],[y1,y2],color='red')
                plt.plot([x1,x2],[y1,y1],color='red')
                plt.plot([x1,x2],[y2,y2],color='red')
                
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
        edge_bw = cv2.drawContours(zeros,[self.boundary],0,(255,255,255),2)
        x_min = self.xmin
        
        x_start = x_min-window_background
        x_end = x_min+window_tape
        edge_bw = edge_bw[:,x_start:x_end]
        if size is not None:
            edge_bw = cv2.resize(edge_bw,(size[1],size[0]))
        if plot:
            plt.figure()
            plt.imshow(edge_bw,cmap='gray')
        return edge_bw
    