#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:15:29 2020

@author: petavazohi
"""
import h5py
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

rf = h5py.File('TapeMatchData.hdf5','r')
data = rf['line'][:][:,:,:,0] # shape= (1370, 4, 1000)
matches = rf['matches'][:]
rf.close()

data = data.reshape(-1,4000)
train_idx = np.random.randint(data.shape[0], size=int(data.shape[0]*8/10))
test_idx = np.random.randint(data.shape[0], size=int(data.shape[0]*2/10))

train_data = data[train_idx]
test_data  = data[test_idx ]
train_labels = matches[train_idx]
test_labels  = matches[train_idx]

train_data = data
train_labels = matches
x_train,x_test,y_train,y_test = train_test_split(train_data,train_labels,test_size=0.3,random_state=10)

model = keras.Sequential([
        keras.layers.Flatten(input_shape=(4000,)),
        keras.layers.Dense(1000,activation=tf.nn.relu),
        keras.layers.Dense(1000,activation=tf.nn.relu),
        keras.layers.Dense(1,activation=tf.nn.sigmoid),])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=25, batch_size=10)