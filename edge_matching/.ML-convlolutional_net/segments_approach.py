#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:23:32 2020

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
#import matplotlib.pylab as plt

def learner(array): # 4 x 30*300 x 1 each image  
        #encoder 
        #in put = 30 x 300 
        img1 = Lambda(lambda x: x[:,0,:,:])(array) 
        img2 = Lambda(lambda x: x[:,1,:,:])(array) 
        img3 = Lambda(lambda x: x[:,2,:,:])(array) 
        img4 = Lambda(lambda x: x[:,3,:,:])(array) 
        print(array.shape) 
        print(img1.shape) 
            
        conv1_1 = Conv2D(16, (3,3), activation='relu', padding='same')(img1) # 30x300 x 16 
        pool1_1 = MaxPooling2D(pool_size=(2,2))(conv1_1) # 15x150x16
        conv2_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1_1) # 15x 150 x 8
        pool2_1 = MaxPooling2D(pool_size=(2,2))(conv2_1) # 7 x 75 x 8
        conv3_1 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool2_1) # 7x 75 x 8
        pool3_1 = MaxPooling2D(pool_size=(2,2))(conv3_1) # 3 x 37 x 8 
#        conv4_1 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool3_1) # 3x37 x 8  
#        pool4_1 = MaxPooling2D(pool_size=(2,2))(conv4_1) # 1x 18 x 8
#        conv5_1 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool4_1) # 93x16 x 8 
#        pool5_1 = MaxPooling2D(pool_size=(2,2))(conv5_1) # 46 x 8  x 8 
#        conv6_1 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool5_1) # 46 x 8 x 8 
#        pool6_1 = MaxPooling2D(pool_size=(2,2))(conv6_1) # 23 x 4  x 8 
        flat_1 = Flatten()(pool3_1)  # 736 x 1 
 
        conv1_2 = Conv2D(16, (3,3), activation='relu', padding='same')(img2) # 1500 x 266 x 16 
        pool1_2 = MaxPooling2D(pool_size=(2,2))(conv1_2) # 750 x 133 x 16 
        conv2_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1_2) # 750 x 133 x 8 
        pool2_2 = MaxPooling2D(pool_size=(2,2))(conv2_2) # 375 x 66 x 8 
        conv3_2 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool2_2) # 375 x 66x 8 
        pool3_2 = MaxPooling2D(pool_size=(2,2))(conv3_2) # 187 x 33 x 8 
#        conv4_2 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool3_2) # 187x  33 x 8  
#        pool4_2 = MaxPooling2D(pool_size=(2,2))(conv4_2) # 93 x 16 x 8 
#        conv5_2 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool4_2) # 187x  33 x 8 
#        pool5_2 = MaxPooling2D(pool_size=(2,2))(conv5_2) # 93 x 16 x 8 
#        conv6_2 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool5_2) # 46 x 8 x 8 
#        pool6_2 = MaxPooling2D(pool_size=(2,2))(conv6_2) # 23 x 4  x 8 
        flat_2 = Flatten()(pool3_2)  # 11904 x 1 
 
        conv1_3 = Conv2D(16, (3,3), activation='relu', padding='same')(img3) # 1500 x 266 x 16 
        pool1_3 = MaxPooling2D(pool_size=(2,2))(conv1_3) # 750 x 133 x 16 
        conv2_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1_3) # 750 x 133 x 8 
        pool2_3 = MaxPooling2D(pool_size=(2,2))(conv2_3) # 375 x 66 x 8 
        conv3_3 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool2_3) # 375 x 66x 8 
        pool3_3 = MaxPooling2D(pool_size=(2,2))(conv3_3) # 187 x 33 x 8 
#        conv4_3 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool3_3) # 187x  33 x 8  
#        pool4_3 = MaxPooling2D(pool_size=(2,2))(conv4_3) # 93 x 16 x 8 
#        conv5_3 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool4_3) # 187x  33 x 8 
#        pool5_3 = MaxPooling2D(pool_size=(2,2))(conv5_3) # 93 x 16 x 8 
#        conv6_3 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool5_3) # 46 x 8 x 8 
#        pool6_3 = MaxPooling2D(pool_size=(2,2))(conv6_3) # 23 x 4  x 8 
        flat_3 = Flatten()(pool3_3)  # 11904 x 1 
  
        conv1_4 = Conv2D(16, (3,3), activation='relu', padding='same')(img4) # 1500 x 266 x 16 
        pool1_4 = MaxPooling2D(pool_size=(2,2))(conv1_4) # 750 x 133 x 16 
        conv2_4 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1_4) # 750 x 133 x 8 
        pool2_4 = MaxPooling2D(pool_size=(2,2))(conv2_4) # 375 x 66 x 8 
        conv3_4 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool2_4) # 375 x 66x 8 
        pool3_4 = MaxPooling2D(pool_size=(2,2))(conv3_4) # 187 x 33 x 8 
#        conv4_4 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool3_4) # 187x  33 x 8  
#        pool4_4 = MaxPooling2D(pool_size=(2,2))(conv4_4) # 93 x 16 x 8 
#        conv5_4 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool4_4) # 187x  33 x 8 
#        pool5_4 = MaxPooling2D(pool_size=(2,2))(conv5_4) # 93 x 16 x 8                                                                                                                              
#        conv6_4 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool5_4) # 46 x 8 x 8 
#        pool6_4 = MaxPooling2D(pool_size=(2,2))(conv6_4) # 23 x 4  x 8 
        flat_4 = Flatten()(pool3_4)  # 11904 x 1 
 
        layer1 = Concatenate()([flat_1,flat_2,flat_3,flat_4]) # 576
        layer2 = Dense(10000,activation='relu')(layer1) # 10000 
        layer3 = Dense(1000,activation='relu')(layer2) # 1000 
        layer4 = Dense(1000,activation='relu')(layer3) # 1000 
        layer5 = Dense(200,activation='relu')(layer4)  # 200 
        layer6 = Dense(1,activation='sigmoid')(layer5) # 1 
        
        return layer6                                                                                                                                                                               
   
   
rf = h5py.File('TapeMatchData.hdf5','r')
matches = rf['matches'][:]
c = 0
data = np.zeros((52746,4,30,300))
labels = np.zeros(52746,)
for iset in range(1,1370):
    for i in range(rf['segments'][str(iset)][:].shape[1]):

        data[c] = rf['segments'][str(iset)][:][:,i,:,:]
        labels[c] = matches[iset]
        c+=1
        
rf.close()
data = data.reshape(-1,4,30,300,1)
#data = data/255.
batch_size = 20 
epochs = 50 
inChannel = 1 
input_img = Input(shape = (4,30,300, inChannel)) 
model = Model(input_img, learner(input_img)) 
#model.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer = RMSprop()) 
model.summary() 

train_X,test_X,train_y,test_y = train_test_split(data, labels,test_size=0.2, random_state=13)
model_train = model.fit(train_X, train_y, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_X, test_y))