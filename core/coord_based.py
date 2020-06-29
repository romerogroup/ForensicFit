# -*- coding: utf-8 -*-

__author__ = 'Pedram Tavadze'

import numpy as np
import matplotlib.pylab as plt


class CoordinateBased:
    """
        
        Parameters
        ----------
        edge : list of floats
            The profile of the edge of the image
        npoints : TYPE, optional
            number of points used to describe the profile in the output

    """
        
    def __init__(self,
                 edge=None,
                 npoints=1000,):
        
        self.edge = edge
        self.npoints = npoints
        
        
        
    @property
    def data(self):
        """
        

        Returns
        -------
        2d numpy array of floats
            Numpy array representing the profile of the edge using the number 
            of the points provided npoints

        """
        data = np.zeros(shape=(self.npoints,2))
        for ipoint in range(self.npoints):
            y_start = self.ymin+ipoint*self.dy
            y_end   = y_start+self.dy
            cond1 = self.edge[:,1] >= y_start
            cond2 = self.edge[:,1] <= y_end
            cond_and = np.bitwise_and(cond1,cond2)
            data[ipoint,:] = np.average(self.edge[cond_and],axis=0)
        return self.data
    
    
    @property
    def ymin(self):
        """
        

        Returns
        -------
        float
            minimum in y direction

        """
        return min(self.edge[:,1])
    
    @property 
    def ymax(self):
        """
        

        Returns
        -------
        float
            maximum in y direction

        """
        return max(self.edge[:,1])
    
    @property
    def dy(self):
        """
        

        Returns
        -------
        float
            distance between two adjacent points in y directions

        """
        return self.y_interval/self.npoints
    
    @property
    def y_interval(self):
        """
        

        Returns
        -------
        float
            The whole distance the profile covers in y direction, ymax-ymin

        """
        return self.ymax - self.ymin
    
    @property
    def xmin(self):
        """
        

        Returns
        -------
        float
            minimum in x direction

        """
        return min(self.edge[:,0])
    
    @property
    def xmax(self):
        """
        

        Returns
        -------
        float
            maximum in y direction

        """
        return max(self.edge[:,0])
    
    def plot_scatter(self):
        """
        This function plots the profile using a scatter plot from matplotlib


        """
        plt.scatter(self.data[:,0],self.data[:,1])
        plt.show()
    
    def plot_profile(self):
        """
        This function plots the profile using a plot from matplotlib

        """
        plt.plot(self.data[:,0],self.data[:,1])
        plt.show()
    