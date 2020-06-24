from PIL import Image
import os
import h5py
import numpy as np


ls = os.listdir('.')
ndata = 174+345+800+204+248
nx = 1500
ny = 1500
window = 200 # make sure it's devisable by 4, the window will be window+window//4

rf = h5py.File('{}x{}.hdf5'.format(nx,window+window//3),'w')          
data = rf.create_dataset('data',(ndata*2,nx,window+window//3))
labels = rf.create_dataset('labels',(ndata*2,),dtype='S10')

c = 0

for idir in ls :
    if not os.path.isdir(idir):
        continue
    ls2 = os.listdir(idir)
    for ifile in ls2 :
        path = "{}{}{}".format(idir,os.sep,ifile) 
        print(path)
        # if c > 100 :
        #     continue
        if '.tif' in ifile and not 'original' in ifile:

            
            img = Image.open(path)
            w,h = img.size
            m = w//2
            img_left = np.array(list(img.resize(size=(nx,ny),box=(0,0,m,h)).getdata())).reshape(nx,ny)
            img_right = np.array(list(img.resize(size=(nx,ny),box=(m,0,w,h)).getdata())).reshape(nx,ny)

            arg_r = np.argmax(img_right.sum(axis=0)<60000)
            start = arg_r-window
            end = arg_r+window//3
            if end > 1999 :
                end = 1999
            img_right = img_right[:,start:end]

            arg_l = np.argmax(img_left.sum(axis=0)>60000)
            start = arg_l-window//3
            end   =   arg_l+window
            if start < 0 :
                start = 0
            img_left = img_left[:,start:end]
            print(arg_r,arg_l)
            data[c,:,:] = img_left
            data[c+1,:,:] = img_right
            labels[c] = np.string_(ifile[:-4]+'_L')
            labels[c+1] = np.string_(ifile[:-4]+'_R')
            c += 2
            
            
            # img.show()
            # img_left.show()
            # img_right.show()
            # img_left.save(fp=ifile[:-4]+'_L.jpg')
            # img_right.save(fp=ifile[:-4]+'_R.jpg')
            # img.save(fp=ifile[:-4]+'.jpg')
            # c+=1
            # np.array(list(img_left.getdata())).reshape(2000,2000))        
        
rf.close()    
