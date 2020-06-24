#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:40:19 2020

@author: petavazohi
"""


# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import h5py
import numpy as np


rf = h5py.File('TapeMatchData.hdf5','r')
data = rf['line'][:][:,:,:,0] # shape= (1370, 4, 1000)
y_ped = rf['matches'][:].astype(np.float64)
rf.close()

#for idata in range(len(data)):
#    for j in range(4):
#        data[idata][j] = data[idata][j]/np.linalg.norm(data[idata][j])
X_ped = data.reshape(-1,4000).round(2)
X_ped[np.isnan(X_ped)] = 0
#X_ped = X_ped[0:768,0:8]
#y_ped = y_ped[0:768]
# define the keras model
model = Sequential()
model.add(Dense(4000, input_dim=4000, activation='relu'))
model.add(Dense(1000, input_dim=4000, activation='relu'))
model.add(Dense(1000, input_dim=4000, activation='relu'))
model.add(Dense(1000, input_dim=4000, activation='relu'))
model.add(Dense(100, input_dim=4000, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

## compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#
#
#...
## fit the keras model on the dataset
model.fit(X_ped, y_ped, epochs=150, batch_size=20)
