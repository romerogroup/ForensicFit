# -*- coding: utf-8
# import tensorflow as tf

from .. import HAS_TENSORFLOW
import numpy as np
from collections.abc import Mapping
from abc import ABCMeta, abstractmethod
import os

"""
TODO :
  imbalearn
"""

class DatasetNumpy:
    def __init__(self, X :np.array, y: np.array, extra={}, name=''):

        self.X = X
        self.y = y
        self.extra = {key:np.array(extra[key]) for key in extra}
        self.values = {"X": self.X, "y": self.y}
        self.metadata = {"mode": "data", 'name': name}
        self.name = name

        self._train_indicies = None
        self._test_indicies = None
        self.shuffle()

    @property
    def shape(self):
        ret = {key:self.extra[key].shape for key in self.extra}
        ret['X'] = self.X.shape
        ret['y'] = self.y.shape
        return ret

        
    def shuffle(self, train_size=0.8):
        indicies = np.random.randint(0, self.ndata, (self.ndata))
        train_length = int(round(0.8*self.ndata,0))
        self._train_indicies = indicies[:train_length]
        self._test_indicies = indicies[train_length:]

        
    def balance(self):
        print("balancing the data")
        indicies = self.indicies
        nclasses = {key:len(indicies[key]) for key in indicies}
        smallest_class = min(nclasses, key=nclasses.get)
        print("class with the smallest data points {} with {} points".format(smallest_class, nclasses[smallest_class]))
        for cl in indicies:
            if cl == smallest_class:
                continue
            else:
                nremove = nclasses[cl] - nclasses[smallest_class]
                print(nremove, cl)
                idxs = np.random.choice(indicies[cl], size=nremove, replace=False)
                # indicies[cl][np.random.randint(nclasses[cl], size=nremove)]
                self.X = np.delete(self.X, idxs, 0)
                self.y = np.delete(self.y, idxs, 0)
                for key in self.extra:
                    self.extra[key] = np.delete(self.extra[key], idxs, 0)
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
        print("files saved at {}".format(filename))
        self.shape

    @classmethod
    def load(cls, filename):
        if ".npy" in filename:
            filename = filename.split(".")[0]
        X = np.load("{}{}X.npy".format(filename, os.sep))
        y = np.load("{}{}y.npy".format(filename, os.sep))
        extra = {}
        for ifile in os.listdir(filename):
            if ifile == "X.npy" or ifile == "y.npy":
                continue
            elif ".npy" in ifile:
                extra[ifile.replace('.npy','')] = np.load("{}{}{}".format(filename, os.sep, ifile))
        cls = DatasetNumpy(X, y, extra=extra, name=filename)
        print('loaded {}'.format(filename))
        cls.shape
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

    @property
    def nclass(self):
        return len(self.classes)

    @property
    def classes(self):
        return np.unique(self.y)

    @property
    def indicies(self):
        ret = {}
        for cl in self.classes:
            ret[cl] = np.where(self.y == cl)[0]
        return ret
    
    def __str__(self):
        shape = self.shape
        to_print = 'Dataset Numpy, {}\n'.format(self.name)
        to_print += '============================\n'
        to_print += 'number of classes {}\n'.format(self.nclass)
        indicies = self.indicies
        for cl in self.classes:
            to_print += 'number of elements in class {} = {}\n'.format(cl, len(indicies[cl]))
        to_print += '============================\n'        
        for key in shape:
            to_print += "shape of {:>18} = {}\n".format(key, str(shape[key]))
        
        return to_print

    def __contains__(self, x):
        return x in self.values

    def __getitem__(self, x):
        return self.values.__getitem__(x)

    def __iter__(self):
        return self.values.__iter__()

    def __len__(self):
        return self.values.__len__()

    def __add__(self, new):
        X = np.append(self.X, new.X, axis=0)
        y = np.append(self.y, new.y, axis=0)
        extra = {}
        for key in self.extra:
            extra[key] = np.append(self.extra[key], new.extra[key], axis=0)    
        name = self.name
        return DatasetNumpy(X, y, extra, name)

if HAS_TENSORFLOW:
    class DatasetTensorFlow(tf.data.Dataset):
        def __init__(self, **kwargs):
            super(DatasetTensorFlow, self).__init__(**kwargs)
            
        def from_excel(filenames, **db_settings):
            db = Database(**db_settings)
        
    
