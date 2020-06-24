#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:50:01 2020

@author: petavazohi
"""

import h5py
import numpy as np
import pandas as pd
#from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Reshape, Concatenate
#from keras.layers import Input
#from keras.models import Model
#
#from keras.optimizers import RMSprop

rf = h5py.File('1500x266.hdf5','r')

data = rf['data'][:]
labels = rf['labels'][:]
df_LQ = pd.read_excel('Automated_Algorithm_Scans.xlsx','Low Quality')
df_MQHT = pd.read_excel('Automated_Algorithm_Scans.xlsx','MQHT')
df_MQSC = pd.read_excel('Automated_Algorithm_Scans.xlsx','MQSC')
df_HQHT_C = pd.read_excel('Automated_Algorithm_Scans.xlsx','HQHT - Set C')
df_HQHT_1 = pd.read_excel('Automated_Algorithm_Scans.xlsx','HQHT - Set 1')
df_matches_MQHT = pd.read_excel('Duct_Tape_Known_Non_Match_all_sets.xlsx','Mid-Quality_Hand_Torn')
df_matches_MQSC = pd.read_excel('Duct_Tape_Known_Non_Match_all_sets.xlsx','Mid-Quality_Scissor_Cut')
df_matches_HQHT = pd.read_excel('Duct_Tape_Known_Non_Match_all_sets.xlsx','High-Quality_Hand_Torn')
df_matches_LQHT = pd.read_excel('Duct_Tape_Known_Non_Match_all_sets.xlsx','Low-Quality_Hand_Torn') 

data_ready2use = []
df = pd.DataFrame([['1','2','3','4']],columns=['file1','file2','file3','file4']) 


print('MQHT')
for i in range(df_matches_MQHT.shape[0]):
    item1 = df_matches_MQHT.iloc[i][0]
    item2 = df_matches_MQHT.iloc[i][1]
    
    tapeNo1 = float(item1[:-1])
    tapeNo2 = float(item2[:-1])
    
    idx1  = df_MQHT['Tape'] == tapeNo1
    idx2  = df_MQHT['Tape'] == tapeNo2
    if df_MQHT[idx1].shape[0] != 2 :
        print(item1,item2,i+2)
        print(df_MQHT[idx1])
        continue
    if df_MQHT[idx2].shape[0] != 2 :
        print(item1,item2,i+2)
        print(df_MQHT[idx2])
        continue
    sub_df1 = df_MQHT[idx1]
    sub_df2 = df_MQHT[idx2]

    matching_labels = []
    for j in range(2):
        if sub_df1.iloc[j]['Left Edge'] == item1[-1]:
            matching_labels.append(sub_df1.iloc[j]['Scan Name']+'_L' )
        else :
            matching_labels.append(sub_df1.iloc[j]['Scan Name']+'_R')
        if sub_df2.iloc[j]['Left Edge'] == item2[-1]:
            matching_labels.append(sub_df2.iloc[j]['Scan Name']+'_L' )
        else :
            matching_labels.append(sub_df2.iloc[j]['Scan Name']+'_R')
    matching_labels = np.string_(matching_labels)
    df_temp = pd.DataFrame([matching_labels],columns=['file1','file2','file3','file4']) 
    df = df.append(df_temp)
   
print('MQSC')                                                    
for i in range(df_matches_MQSC.shape[0]):
    item1 = df_matches_MQSC.iloc[i][0]
    item2 = df_matches_MQSC.iloc[i][1]
    
    tapeNo1 = float(item1[:-1])
    tapeNo2 = float(item2[:-1])
    idx1  = df_MQSC['Tape'] == tapeNo1
    idx2  = df_MQSC['Tape'] == tapeNo2
    if df_MQSC[idx1].shape[0] != 2 :
        print(item1,item2,i+2)
        print(df_MQSC[idx1])
        continue
    if df_MQSC[idx2].shape[0] != 2 :
        print(item1,item2,i+2)
        print(df_MQSC[idx2])
        continue
    sub_df1 = df_MQSC[idx1]
    sub_df2 = df_MQSC[idx2]

    matching_labels = []
    for j in range(2):
        if sub_df1.iloc[j]['Left Edge'] == item1[-1]:
            matching_labels.append(sub_df1.iloc[j]['Scan Name']+'_L' )
        else :
            matching_labels.append(sub_df1.iloc[j]['Scan Name']+'_R')
        if sub_df2.iloc[j]['Left Edge'] == item2[-1]:
            matching_labels.append(sub_df2.iloc[j]['Scan Name']+'_L' )
        else :
            matching_labels.append(sub_df2.iloc[j]['Scan Name']+'_R')
    matching_labels = np.string_(matching_labels)
    
    df_temp = pd.DataFrame([matching_labels],columns=['file1','file2','file3','file4'])  
    df = df.append(df_temp)
   
print('LQHT')                                                    
for i in range(df_matches_LQHT.shape[0]):
    item1 = df_matches_LQHT.iloc[i][0]
    item2 = df_matches_LQHT.iloc[i][1]

    tapeNo1 = float(item1[:-1])
    tapeNo2 = float(item2[:-1])
    idx1  = df == tapeNo1
    idx2  = df_LQ['Tape'] == tapeNo2
    if df_LQ[idx1].shape[0] != 2 :
        print(item1,item2,i+2)
        print(df_LQ[idx1])
    if df_LQ[idx2].shape[0] != 2 :
        print(item1,item2,i+2)
        print(df_LQ[idx2])
    sub_df1 = df_LQ[idx1]
    sub_df2 = df_LQ[idx2]

    matching_labels = []
    for j in range(2):
        if sub_df1.iloc[j]['Obvious Cut'] == 'Left':
            matching_labels.append(sub_df1.iloc[j]['Scan Name']+'_L' )
        else :
            matching_labels.append(sub_df1.iloc[j]['Scan Name']+'_R')
        if sub_df2.iloc[j]['Obvious Cut'] == 'Left':
            matching_labels.append(sub_df2.iloc[j]['Scan Name']+'_L' )
        else :
            matching_labels.append(sub_df2.iloc[j]['Scan Name']+'_R')
    matching_labels = np.string_(matching_labels)
    df_temp = pd.DataFrame([matching_labels],columns=['file1','file2','file3','file4']) 
    df = df.append(df_temp)


print('HQHT')                                                                                                                            
for i in range(df_matches_HQHT.shape[0]):
    item1 = df_matches_HQHT.iloc[i][0]
    item2 = df_matches_HQHT.iloc[i][1]
    tapeNo1 = float(item1[:-1])
    tapeNo2 = float(item2[:-1])
    idx1_1  = df_HQHT_1['Tape'] == tapeNo1
    idx2_1  = df_HQHT_1['Tape'] == tapeNo2
    idx1_C  = df_HQHT_C['Tape'] == tapeNo1
    idx2_C  = df_HQHT_C['Tape'] == tapeNo2
    if df_HQHT_1[idx1_1].shape[0] == 2 and df_HQHT_1[idx2_1].shape[0] == 2:
        sub_df1 = df_HQHT_1[idx1_1]
        sub_df2 = df_HQHT_1[idx2_1]
        matching_labels = []
        for j in range(2):
            if sub_df1.iloc[j]['Left Edge'] == item1[-1]:
                matching_labels.append(sub_df1.iloc[j]['Scan Name']+'_L' )
            else :
                matching_labels.append(sub_df1.iloc[j]['Scan Name']+'_R')
            if sub_df2.iloc[j]['Left Edge'] == item2[-1]:
                matching_labels.append(sub_df2.iloc[j]['Scan Name']+'_L' )
            else :
                matching_labels.append(sub_df2.iloc[j]['Scan Name']+'_R')
        matching_labels = np.string_(matching_labels)
        df_temp = pd.DataFrame([matching_labels],columns=['file1','file2','file3','file4'])    
        df = df.append(df_temp)
        
        
    elif df_HQHT_C[idx1_C].shape[0] == 2 or df_HQHT_C[idx2_C].shape[0] == 2:
        sub_df1 = df_HQHT_C[idx1_C]
        sub_df2 = df_HQHT_C[idx2_C]
        matching_labels = []
        for j in range(2):
            if sub_df1.iloc[j]['Obvious Cut'] == 'Left':
                matching_labels.append(sub_df1.iloc[j]['Scan Name']+'_R' )
            else :
                matching_labels.append(sub_df1.iloc[j]['Scan Name']+'_L')
            if sub_df2.iloc[j]['Obvious Cut'] == 'Left':
                matching_labels.append(sub_df2.iloc[j]['Scan Name']+'_R' )
            else :
                matching_labels.append(sub_df2.iloc[j]['Scan Name']+'_L')
        matching_labels = np.string_(matching_labels)
        df_temp = pd.DataFrame([matching_labels],columns=['file1','file2','file3','file4'])    
        df = df.append(df_temp)
        
    else :
        print(item1,item2,i+2)            
        continue
