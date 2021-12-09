# -*- coding: utf-8 -*-

import cv2
from matplotlib import pylab as plt
import numpy as np
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping


class Analyzer:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.label = None
        self.image = None
        self.mode = 'analysis'
        self.material = None
        self.boundary = None
        self.values = {}
        self.metadata = {'mode': 'analysis'}

    def plot_boundary(self, savefig=None, color='r', ax=None, show=False):
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
        if ax is None:
            plt.figure()
            ax = plt.subplot(111)
        ax.plot(self.boundary[:, 0], self.boundary[:, 1], c=color)
        if savefig is not None:
            plt.savefig(savefig)
        elif show:
            plt.show()

        
    def plot(self, which, cmap='viridis', savefig=None, ax=None, reverse_x=False):
        
        if which == "coordinate_based":
            if ax is None:
                plt.figure(figsize=(2,8))
                ax = plt.subplot(111)
                
            # ax.scatter(self[which]['means'][:, 0],
            #            np.flip(self[which]['means'][:, 1]),
            #            c='red',
            #            s=1)
            ax.errorbar(self[which]['means'][:, 0],
                        np.flip(self[which]['means'][:, 1]),
                        xerr= self[which]['stds'],
                        ecolor='blue',
                        color='red',
                        markersize=0.5,
                        fmt='o')
            ax.set_ylim(min(self[which]['means'][:, 1]),max(self[which]['means'][:, 1]))
            if reverse_x :
                ax.set_xlim(max(self[which]['means'][:, 0])*1.1, min(self[which]['means'][:, 0])*0.9)
            else :
                ax.set_xlim(min(self[which]['means'][:, 0])*0.9, max(self[which]['means'][:, 0])*1.1)
        elif len(self[which].shape) > 2:
            if ax is None:
                plt.figure()
                ax = plt.subplot(111)
            dynamic_positions = self.metadata['analysis'][which]['dynamic_positions']
            image = self.image.copy()

            xs = []
            for iseg in dynamic_positions:
                y1 = iseg[1][0]
                y2 = iseg[1][1]
                x1 = iseg[0][0]
                x2 = iseg[0][1]
                xs.append(x1)
                xs.append(x2)
                ax.plot([x1, x1], [y1, y2], color='red')
                ax.plot([x2, x2], [y1, y2], color='red')
                ax.plot([x1, x2], [y1, y1], color='red')
                ax.plot([x1, x2], [y2, y2], color='red')
                image[y1:y2, :x1]=0
                image[y1:y2, x2:]=0
            ax.imshow(image, cmap=cmap)
            if reverse_x:
                ax.set_xlim(max(xs)*1.1, min(xs)*0.9)
            else:
                ax.set_xlim(min(xs)*0.9, max(xs)*1.1)
        else:
            if ax is None:
                plt.figure()
                ax = plt.subplot(111)
            ax.imshow(self[which], cmap=cmap)
            if reverse_x :
                ax.set_xlim(self[which].shape[1], 0)
            else :
                ax.set_xlim(0, self[which].shape[1])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        if savefig is not None:
            ax.savefig(savefig)

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
