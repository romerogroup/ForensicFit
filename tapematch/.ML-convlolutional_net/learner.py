import h5py
import numpy as np
import pandas as pd
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Reshape, Concatenate
from keras.layers import Input
from keras.models import Model

from keras.optimizers import RMSprop

rf = h5py.File('1500x266.hdf5','r')

data = rf['data'][:]
labels = rf['labels'][:]
df_LQ = pd.read_excel('Automated_Algorithm_Scans.xlsx','Low Quality')
df_MQHT = pd.read_excel('Automated_Algorithm_Scans.xlsx','MQHT')
df_MQSC = pd.read_excel('Automated_Algorithm_Scans.xlsx','MQSC')
df_HQHT_C = pd.read_excel('Automated_Algorithm_Scans.xlsx','HQHT - Set C')
df_HQHT_1 = pd.read_excel('Automated_Algorithm_Scans.xlsx','HQHT - Set 1')
df_matches_MQHT = pd.read_excel('Duct_Tape_True_Matches_all_sets.xlsx','Mid-Quality_Hand_Torn')
df_matches_MQSC = pd.read_excel('Duct_Tape_True_Matches_all_sets.xlsx','Mid-Quality_Scissor_Cut')
df_matches_HQHT = pd.read_excel('Duct_Tape_True_Matches_all_sets.xlsx','High-Quality_Hand_Torn')
df_matches_LQHT = pd.read_excel('Duct_Tape_True_Matches_all_sets.xlsx','Low-Quality_Hand_Torn') 

data_ready2use = []

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
    data_matched = []
    for ilabel in matching_labels :
        data_matched.append(data[labels == ilabel][0])
    data_ready2use.append(np.array(data_matched))
        
                                                    
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
    data_matched = []
    for ilabel in matching_labels :
        data_matched.append(data[labels == ilabel][0])
    data_ready2use.append(np.array(data_matched))


for i in range(df_matches_LQHT.shape[0]):
    item1 = df_matches_LQHT.iloc[i][0]
    item2 = df_matches_LQHT.iloc[i][1]

    tapeNo1 = float(item1[:-1])
    tapeNo2 = float(item2[:-1])
    idx1  = df_LQ['Tape'] == tapeNo1
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

    data_matched = []
    for ilabel in matching_labels :
        data_matched.append(data[labels == ilabel][0])
    data_ready2use.append(np.array(data_matched))
                            
                                                                        
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
        data_matched = []
        for ilabel in matching_labels :
            data_matched.append(data[labels == ilabel][0])
        data_ready2use.append(np.array(data_matched))
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

        data_matched = []
        for ilabel in matching_labels :
            data_matched.append(data[labels == ilabel][0])
        data_ready2use.append(np.array(data_matched))
    else :
        print(item1,item2,i+2)            
        continue
    
ntrues = len(data_ready2use)
nfalse = ntrues

matches = np.zeros(shape=(ntrues+nfalse,))
matches[0:ntrues] = np.ones(shape=(ntrues))



for i in range(nfalse):
    idx = np.random.randint(0,len(data),4)
    data_ready2use.append(data[idx])

data_ready2use = np.array(data_ready2use)
                                                            
# data_ready2use.shape  (396, 4, 1500, 266)


def learner([img1,img2,img3,img4]): # 4 x 1500 x 266 x 1 each image 
        #encoder
        #in put = 1500 x 266 x 2
        conv1_1 = Conv2D(16, (3,3), activation='relu', padding='same')(img1) # 1500 x 266 x 16
        pool1_1 = MaxPooling2D(pool_size=(2,2))(conv1_1) # 750 x 133 x 16
        conv2_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1_1) # 750 x 133 x 8
        pool2_1 = MaxPooling2D(pool_size=(2,2))(conv2_1) # 375 x 66 x 8
        conv3_1 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool2_1) # 375 x 66x 8
        pool3_1 = MaxPooling2D(pool_size=(2,2))(conv3_1) # 187 x 33 x 8
        conv4_1 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool3_1) # 187x  33 x 8 
        pool4_1 = MaxPooling2D(pool_size=(2,2))(conv4_1) # 93 x 16 x 8
        conv5_1 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool4_1) # 93x16 x 8
        pool5_1 = MaxPooling2D(pool_size=(2,2))(conv5_1) # 46 x 8  x 8
        conv6_1 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool5_1) # 46 x 8 x 8
        pool6_1 = MaxPooling2D(pool_size=(2,2))(conv6_1) # 23 x 4  x 8
        flat_1 = Flatten()(pool6_1)  # 736 x 1

        conv1_2 = Conv2D(16, (3,3), activation='relu', padding='same')(img2) # 1500 x 266 x 16
        pool1_2 = MaxPooling2D(pool_size=(2,2))(conv1_2) # 750 x 133 x 16
        conv2_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1_2) # 750 x 133 x 8
        pool2_2 = MaxPooling2D(pool_size=(2,2))(conv2_2) # 375 x 66 x 8
        conv3_2 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool2_2) # 375 x 66x 8
        pool3_2 = MaxPooling2D(pool_size=(2,2))(conv3_2) # 187 x 33 x 8
        conv4_2 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool3_2) # 187x  33 x 8 
        pool4_2 = MaxPooling2D(pool_size=(2,2))(conv4_2) # 93 x 16 x 8
        conv5_2 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool4_2) # 187x  33 x 8
        pool5_2 = MaxPooling2D(pool_size=(2,2))(conv5_2) # 93 x 16 x 8
        conv6_2 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool5_2) # 46 x 8 x 8
        pool6_2 = MaxPooling2D(pool_size=(2,2))(conv6_2) # 23 x 4  x 8
        flat_2 = Flatten()(pool6_2)  # 11904 x 1

        conv1_3 = Conv2D(16, (3,3), activation='relu', padding='same')(img3) # 1500 x 266 x 16
        pool1_3 = MaxPooling2D(pool_size=(2,2))(conv1_3) # 750 x 133 x 16
        conv2_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1_3) # 750 x 133 x 8
        pool2_3 = MaxPooling2D(pool_size=(2,2))(conv2_3) # 375 x 66 x 8
        conv3_3 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool2_3) # 375 x 66x 8
        pool3_3 = MaxPooling2D(pool_size=(2,2))(conv3_3) # 187 x 33 x 8
        conv4_3 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool3_3) # 187x  33 x 8 
        pool4_3 = MaxPooling2D(pool_size=(2,2))(conv4_3) # 93 x 16 x 8
        conv5_3 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool4_3) # 187x  33 x 8
        pool5_3 = MaxPooling2D(pool_size=(2,2))(conv5_3) # 93 x 16 x 8
        conv6_3 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool5_3) # 46 x 8 x 8
        pool6_3 = MaxPooling2D(pool_size=(2,2))(conv6_3) # 23 x 4  x 8
        flat_3 = Flatten()(pool6_3)  # 11904 x 1

        conv1_4 = Conv2D(16, (3,3), activation='relu', padding='same')(img4) # 1500 x 266 x 16
        pool1_4 = MaxPooling2D(pool_size=(2,2))(conv1_4) # 750 x 133 x 16
        conv2_4 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1_4) # 750 x 133 x 8
        pool2_4 = MaxPooling2D(pool_size=(2,2))(conv2_4) # 375 x 66 x 8
        conv3_4 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool2_4) # 375 x 66x 8
        pool3_4 = MaxPooling2D(pool_size=(2,2))(conv3_4) # 187 x 33 x 8
        conv4_4 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool3_4) # 187x  33 x 8 
        pool4_4 = MaxPooling2D(pool_size=(2,2))(conv4_4) # 93 x 16 x 8
        conv5_4 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool4_4) # 187x  33 x 8
        pool5_4 = MaxPooling2D(pool_size=(2,2))(conv5_4) # 93 x 16 x 8
        conv6_4 = Conv2D(8, ( 3,3), activation='relu', padding='same')(pool5_4) # 46 x 8 x 8
        pool6_4 = MaxPooling2D(pool_size=(2,2))(conv6_4) # 23 x 4  x 8
        flat_4 = Flatten()(pool6_4)  # 11904 x 1

        layer1 = Concatenate()([flat_1,flat_2,flat_3,flat_4]) # 47616
        layer2 = Dense(1000,activation='relu')(layer1) # 10000
        layer3 = Dense(1000,activation='relu')(layer2) # 1000
        layer4 = Dense(1000,activation='relu')(layer3) # 1000
        layer5 = Dense(200,activation='relu')(layer4)  # 200
        layer6 = Dense(1,activation='sigmoid')(layer5) # 1
        
        return layer6
                                                                
data_ready2use = data_ready2use.reshape(798,4,1500,266,1)


batch_size = 20
epochs = 50
inChannel = 1
input_img = Input(shape = (4,1500,266, inChannel))
model = Model(input_img, learner(input_img))
model.compile(loss='mean_squared_error', optimizer = RMSprop())
model.summary()

from sklearn.model_selection import train_test_split
train_X_data,valid_X_data,train_ground_matches,valid_ground_matches = train_test_split(data_ready2use, matches,test_size=0.2, random_state=13) 
autoencoder_train = model.fit(train_X_data, train_ground_matches, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X_data, valid_ground_matches))

