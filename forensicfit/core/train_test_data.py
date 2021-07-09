# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from collections.abc import Mapping
from abc import ABCMeta, abstractmethod


class TrainTestData(Mapping):
    __metaclass__ = ABCMeta

    def __init__(
        self,
        data,
        test_size=0.2,
        train_size=None,
        random_state=None,
        shuffle=True,
        stratify=None,
    ):
        self.values = {}
        self.metadata = {"mode": "train_test_data"}
        self.data = data
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify
        X_train, X_test, y_train, y_test = train_test_split(
            data.X, data.y, 
            test_size=test_size, 
            train_size=train_size, 
            random_state=random_state, 
            shuffle=shuffle, 
            stratify=stratify,
        )
        self.train = Data(X_train, y_train)
        self.test = Data(X_test, y_test)
        self.load_dict()
        self.load_metadata()

    def load_dict(self):
        self.values["train"] = self.train.values
        self.values["test"] = self.test.values
        self.values["data"] = self.data.values
        

    def load_metadata(self):
        self.metadata["test_size"] = self.test_size
        self.metadata["train_size"] = self.train_size
        self.metadata["random_state"] = self.random_state
        self.metadata["shuffle"] = self.shuffle
        self.metadata["stratify"] = self.stratify

    @classmethod
    def from_dict(cls, values, metadata):
        if values is None:
            raise Exception(
                "The provided dictionary was empty. Maybe change the query criteria"
            )
        cls = TrainTestData()
        for key in values:
            if not isinstance(getattr(type(cls), key, None), property):
                setattr(cls, key, values[key])
        cls.data = values["data"]
        cls.train = Data(values["train"]["X"], values["train"]["y"])
        cls.test = Data(values["test"]["X"], values["test"]["y"])
        cls.load_dict()
        cls.metadata = metadata
        return cls

    def redraw(
        self,
        test_size=0.2,
        train_size=None,
        random_state=None,
        shuffle=True,
        stratify=None,
    ):
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify
        X_train, X_test, y_train, y_test = train_test_split(
            self.data,
            self.labels,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
        self.train = Data(X_train, y_train)
        self.test = Data(X_test, y_test)
        self.load_dict()
        self.load_metadata()

    def __contains__(self, x):
        return x in self.values

    def __getitem__(self, x):
        return self.values.__getitem__(x)

    def __iter__(self):
        return self.values.__iter__()

    def __len__(self):
        return self.values.__len__()


class Data(Mapping):
    __metaclass__ = ABCMeta

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.values = {"X": X, "y": y}
        self.metadata = {"mode": "data"}

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
