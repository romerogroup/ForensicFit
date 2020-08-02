#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:44:44 2020

@author: petavazohi
"""
import h5py
import numpy as np
import pandas as pd
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Reshape, Concatenate
from keras.layers import Input
from keras.models import Model
from keras.optimizers import RMSprop

def learner(input_data): # 1000
        
        layer1 = Dense(1000,activation='relu')(input_data) # 10000
        layer2 = Dense(1000,activation='relu')(layer1) # 1000
        layer3 = Dense(1000,activation='relu')(layer2) # 1000
        layer4 = Dense(200,activation='relu')(layer3)  # 200
        layer5 = Dense(1,activation='sigmoid')(layer4) # 1
        
        return layer5
    
#ls
#data = rf