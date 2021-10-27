# -*- coding: utf-8
import tensorflow as tf
import numpy as np
from collections.abc import Mapping
from abc import ABCMeta, abstractmethod
import os

class DatasetNumpy:
    def __init__(self, X=None, y=None, extra={}, name=''):
        self.X = X
        self.y = y
        self.extra = {key:np.array(extra[key]) for key in extra}
        self.values = {"X": X, "y": y}
        self.metadata = {"mode": "data", 'name': name}
        self.name = name

        self._train_indicies = None
        self._test_indicies = None
        self.shuffle()

        
    def shuffle(self, train_size=0.8):
        indicies = np.random.randint(0, self.ndata, (self.ndata))
        train_length = int(round(0.8*self.ndata,0))
        self._train_indicies = indicies[:train_length]
        self._test_indicies = indicies[train_length:]


    def balance(self):
        matches = np.sum(len(self.y) == 1)
        non_matches = len(self.y) - matches
        more = abs(non_matches - matches)
        if non_matches > matches:
            culprit = 0
        else :
            culprit = 1
        idxs = np.where(self.y == culprit)
        idxs = idxs[np.random.randint(len(idxs), size=more//2)]
        
        self.X = np.delete(self.X, idxs, 0)
        self.y = np.delete(self.y, idxs, 0)
        self.shuffle()
        
    @property
    def train_X(self):
        return self.X[self._train_indicies]
        
    @property
    def train_y(self):
        return self.y[self._train_indicies]

    @property
    def test_X(self):
        return self.X[self._test_indicies]

    @property
    def test_y(self):
        return self.y[self._test_indicies]
    
    def save(self, filename=''):
        if ".npy" in filename:
            filename = filename.split(".")[0]
        if len(filename) == 0:
            filename = self.name
        i = 1
        dir_name = filename
        while os.path.isdir(dir_name):
            dir_name = filename+str(i)
            i += 1
        filename = dir_name
        os.mkdir(filename)
        np.save("{}{}X.npy".format(filename, os.sep), self.X)
        np.save("{}{}y.npy".format(filename, os.sep), self.y)
        for key in self.extra:
            np.save("{}{}{}.npy".format(filename, os.sep, key), self.extra[key])
        print("file {} saved".format(filename))

    @classmethod
    def load(cls, filename=''):
        if ".npy" in filename:
            filename = filename.split(".")[0]
        X = np.load("{}{}X.npy".format(filename, os.sep))
        y = np.load("{}{}y.npy".format(filename, os.sep))
        extra = {}
        for ifile in os.listdir(filename):
            if ifile == "X.npy" or ifile == "y.npy":
                continue
            elif ".npy" in ifile:
                extra[ifile] = np.load("{}{}{}".format(filename, os.sep, ifile))
        cls = DatasetNumpy(X, y, extra=extra, name=filename)
        return cls

    @property
    def ndata(self):
        if self.X is not None:
            return len(self.X)
        else:
            return 0

    @property
    def inputs(self):
        return self.X

    @property
    def outputs(self):
        return self.y

    def __contains__(self, x):
        return x in self.values

    def __getitem__(self, x):
        return self.values.__getitem__(x)

    def __iter__(self):
        return self.values.__iter__()

    def __len__(self):
        return self.values.__len__()

    def __add__(self, new):
        if self.X is None:
            X = new.X
            y = new.y
            extra = new.extra
        elif new.X is None or new.X.shape[0] == 0 :
            X = self.X
            y = self.y
            extra = self.extra
        else:
            X = np.append(self.X, new.X, axis=0)
            y = np.append(self.y, new.y, axis=0)
            extra = {}
            for key in self.extra:
                extra[key] = np.append(self.extra[key], new.extra[key], axis=0)
                
                
        name = self.name
        return DatasetNumpy(X, y, extra, name)


