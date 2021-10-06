# -*- coding: utf-8
import tensorflow as tf
import numpy as np
from collections.abc import Mapping
from abc import ABCMeta, abstractmethod


class DatasetNumpy:
    def __init__(self, X=None, y=None, name=''):
        self.X = X
        self.y = y
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
        np.save("{}_X.npy".format(filename), self.X)
        np.save("{}_y.npy".format(filename), self.y)
        print("file {} saved".format(filename))
    @property
    def ndata(self):
        if self.X is not None:
            return len(self.X)
        else:
            return 0

    @classmethod
    def load(cls, filename=''):
        if ".npy" in filename:
            filename = filename.split(".")[0]
        X = np.load("{}_X.npy".format(filename))
        y = np.load("{}_y.npy".format(filename))
        cls = DatasetNumpy(X, y, name=filename)
        return cls

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
        elif new.X is None or new.X.shape[0] == 0 :
            X = self.X
            y = self.y
        else:
            X = np.append(self.X, new.X, axis=0)
            y = np.append(self.y, new.y, axis=0)
        name = self.name
        return DatasetNumpy(X, y, name)


