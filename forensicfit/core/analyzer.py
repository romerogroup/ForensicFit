# -*- coding: utf-8 -*-

import cv2
from matplotlib import pylab as plt

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
import time

class Analyzer:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.label = None
        self.image = None
        self.mode = 'analysis'
        self.boundary = None
        self.values = {}
        self.metadata = {'mode': 'analysis', 'time':time.asctime()}

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
        if savefig is not None:
            plt.savefig(savefig)
        plt.plot(self.boundary[:, 0], self.boundary[:, 1], c=color)

    def plot(self, which, cmap='viridis', savefig=None):
        plt.figure()
        if len(self[which].shape) > 2:
            dynamic_positions = self.metadata['analysis'][which]['dynamic_positions']

            plt.imshow(self.image, cmap=cmap)
            for iseg in dynamic_positions:
                y1 = iseg[1][0]
                y2 = iseg[1][1]
                x1 = iseg[0][0]
                x2 = iseg[0][1]
                plt.plot([x1, x1], [y1, y2], color='red')
                plt.plot([x2, x2], [y1, y2], color='red')
                plt.plot([x1, x2], [y1, y1], color='red')
                plt.plot([x1, x2], [y2, y2], color='red')
        else:
            plt.imshow(self[which], cmap=cmap)
        if savefig is not None:
            plt.savefig(savefig)

    def add_metadata(self, key, value):
        self.metadata[key] = value

    def show(self, which, wait=0):
        cv2.imshow(which, self[which])
        cv2.waitKey(wait)
        cv2.destroyAllWindows()

    @abstractmethod
    def load_dict(self):
        pass

    @abstractmethod
    def from_dict(self):
        pass

    def __contains__(self, x):
        return x in self.values

    def __getitem__(self, x):
        return self.values.__getitem__(x)

    def __iter__(self):
        return self.values.__iter__()

    def __len__(self):
        return self.values.__len__()
