#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:44:44 2020

@author: petavazohi
"""
import h5py
import numpy as np
#import pandas as pd
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Reshape, Concatenate
from keras.layers import Input
from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import RMSprop,SGD
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt


def learner(array): # 1000
#        print(array.shape)
        
#        img1 = Lambda(lambda x: x[:,0])(array)
#        img2 = Lambda(lambda x: x[:,1])(array)
#        img3 = Lambda(lambda x: x[:,2])(array)
#        img4 = Lambda(lambda x: x[:,3])(array)
        
#        layer1 = Concatenate()([img1,img2,img3,img4])
        layer1 = array
        layer2 = Dense(4000,activation='relu')(layer1) # 10000 
        layer3 = Dense(4000,activation='relu')(layer2)
        layer4 = Dense(4000,activation='relu')(layer3)
        layer5 = Dense(2000,activation='relu')(layer4) # 1000
        layer6 = Dense(1000,activation='tanh')(layer5) # 1000
        layer7 = Dense(1000,activation='relu')(layer6)  # 200
##        layer5 = layer2
        layer8 = Dense(1,activation='sigmoid')(layer7) # 1
#        out = Lambda(lambda layer6: round(layer6))(layer6)
        return layer8



rf = h5py.File('TapeMatchData.hdf5','r')
data = rf['line'][:][:,:,:,0] # shape= (1370, 4, 1000)
matches = rf['matches'][:]
rf.close()


#pp_data = np.zeros((1370,4000))

#for idata in range(len(data)):
#    for j in range(4):
#        data[idata][j] = data[idata][j]/np.linalg.norm(data[idata][j])

data = data.reshape(-1,4000)#[0:798]
#matches = matches[0:798]
col_mean = np.nanmean(data, axis = 0) 
inds = np.where(np.isnan(data)) 
data[inds] = np.take(col_mean, inds[1]) 

train_idx = np.random.randint(data.shape[0], size=int(data.shape[0]*8/10))
test_idx = np.random.randint(data.shape[0], size=int(data.shape[0]*2/10))

#train_data = data[train_idx]
#test_data  = data[test_idx ]
#
#train_labels = matches[train_idx]
#test_labels  = matches[train_idx]

train_data = data
train_labels = matches

batch_size = 10
epochs = 150
inChannel = 1
input_img = Input(shape = (4000,))
model = Model(input_img, learner(input_img))

#model.compile(SGD(lr=0.002, momentum=1.0, decay=0.0, nesterov=True), loss='mse', metrics=['accuracy'])

model.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,
                                                                  train_labels,
                                                                  test_size=0.2)
#                                                                  )

learner_train = model.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))

