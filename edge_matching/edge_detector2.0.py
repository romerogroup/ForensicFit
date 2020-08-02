# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 20:40:17 2020

@author: ptava
"""
import os
import cv2
import numpy as np
import matplotlib.pylab as plt
import imutils
import pandas as pd
import h5py

def get_data(fname,side,rot):
        
        nsegments = {"LQ": 32, "MQ" : 37,"HQ": 55}
        window_tape = 25
        window_black = 25
        window_tape_4p = 100
        window_black_4p = 50
#        size= (30,30)
#        size= (15,30)
#        size_4p = (40,100)
        
#        side = sides[iimage]
#        rot = rotations[iimage]
        
        if 'LQ' in fname :
            Q = 'LQ'
        elif 'MQ' in fname :
            Q = "MQ"
        else : 
            Q = "HQ"
        
        imgray = cv2.imread('all'+os.sep+fname,0)
        imgray = cv2.resize(imgray,(imgray.shape[1]//4,imgray.shape[0]//4))
        imgray = cv2.GaussianBlur(imgray,(5,5),0)
#        cv2.imshow('test',imgray)
        w,h = imgray.shape
        m = h//2
        if 'LQ' in  fname:
            if side == 'L':
                side = 'R'
            else: 
                side = 'L'
                    
        if side == 'L':
            image = imgray[:, 0:m]
        else : 
            image = cv2.flip(imgray[:,m:h],1)
        if rot :
            image = cv2.flip(image,0)
        original = image.copy()
        im_color = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        mask = cv2.inRange(image,60,255)
        
        res = cv2.bitwise_and(image,image,mask=mask)
        
        
      
        contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        max_con = 0
        max_con_arg = 0
        for ic in range(len(contours)):
            if cv2.contourArea(contours[ic]) > max_con :
                max_con = cv2.contourArea(contours[ic])
                max_con_arg = ic
#        cv2.drawContours(im_color,contours,max_con_arg,(0,0,255),2)
#        cv2.imshow(fname+'i',cv2.resize(im_color,tuple(np.array(im_color.shape)[0:2]//3)))
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()  
        boundary = contours[max_con_arg].reshape(contours[max_con_arg].shape[0],2)
        
        
        y_min = boundary[:,1].min()
        y_max = boundary[:,1].max()
        
        x_min = boundary[:,0].min()
        x_max = boundary[:,0].max()
        
        
        ndivision = 5
        
        stds = np.ones((ndivision,2))*1000
        conditions_top = []
        conditions_bottom = []
        for idivision in range(1,ndivision):

            y_interval = y_max-y_min
            cond1 = boundary[:,1]> y_max-y_interval/ndivision
            cond2 = boundary[:,1]< y_max+y_interval/ndivision
            
            x_interal = x_max-x_min
            cond3 = boundary[:,0]>= x_min+x_interal/ndivision*idivision
            cond4 = boundary[:,0]<= x_min+x_interal/ndivision*(idivision+1)
            
            cond_12 = np.bitwise_and(cond1,cond2)
            cond_34 = np.bitwise_and(cond3,cond4)
            cond_and_top = np.bitwise_and(cond_12,cond_34)
            
            # This part is to rotate the images

            if sum(cond_and_top) == 0 :
                conditions_top.append([])
                
                continue
            
            m_top,b0_top = np.polyfit(boundary[cond_and_top][:,0],boundary[cond_and_top][:,1],1)
            std_top = np.std(boundary[cond_and_top][:,1])
            stds[idivision,0] = std_top
            conditions_top.append(cond_and_top)
            
        for idivision in range(1,ndivision):
            
            cond1 = boundary[:,1]> y_min-y_interval/ndivision
            cond2 = boundary[:,1]< y_min+y_interval/ndivision
            
            x_interal = x_max-x_min
            cond3 = boundary[:,0]>= x_min+x_interal/ndivision*idivision
            cond4 = boundary[:,0]<= x_min+x_interal/ndivision*(idivision+1)
            
            cond_12 = np.bitwise_and(cond1,cond2)
            cond_34 = np.bitwise_and(cond3,cond4)
            cond_and_bottom = np.bitwise_and(cond_12,cond_34)
            
            if sum(cond_and_bottom) == 0 :
                
                conditions_bottom.append([])
                continue
            
            
            m_bottom,b0_bottom = np.polyfit(boundary[cond_and_bottom][:,0],boundary[cond_and_bottom][:,1],1)
            
            m = np.average([m_top,m_bottom])
            
            std_bottom = np.std(boundary[cond_and_bottom][:,1])
            # print(std_top,std_bottom)
            
            stds[idivision,1] = std_bottom
            conditions_bottom.append(cond_and_bottom)
#            plt.figure()
#            plt.plot(boundary[:,0],boundary[:,1])
#            plt.scatter(boundary[cond_and_top][:,0],boundary[cond_and_top][:,1])
#            plt.scatter(boundary[cond_and_bottom][:,0],boundary[cond_and_bottom][:,1])

        arg_mins = np.argmin(stds,axis=0)
        cond_and_top = conditions_top[arg_mins[0]]
        cond_and_bottom = conditions_bottom[arg_mins[1]]
        
        crop_y_top = np.average(boundary[cond_and_top][:,1])
        crop_y_bottom = np.average(boundary[cond_and_bottom][:,1])
        
        if np.min(stds,axis=0)[0] > 10 :
            crop_y_top = y_max
        if np.min(stds,axis=0)[1] > 10 :
            crop_y_bottom = y_min
#        angle = np.arctan(m)
        print(imatch,fname,np.min(stds,axis=0)[0],np.min(stds,axis=0)[1])
#        rotated = imutils.rotate(im_color,np.rad2deg(angle))
        rotated = im_color
        

        ##################################################################
        # do the exact same thing on the rotated image
        im_color = rotated[int(crop_y_bottom):int(crop_y_top),:]
#        im_color = cv2.resize(im_color,(1337,1000))
        #im_color = cv2.cvtColor(imgray_resize,cv2.COLOR_GRAY2BGR )
        imgray_resize = cv2.cvtColor(im_color,cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(imgray_resize,60,255)
#        res = cv2.bitwise_and(imgray_resize,imgray_resize,mask=mask)

        
        
        contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        max_con = 0
        max_con_arg = 0
        for ic in range(len(contours)):
            if cv2.contourArea(contours[ic]) > max_con :
                max_con = cv2.contourArea(contours[ic])
                max_con_arg = ic
        zeros = np.zeros_like(im_color)        
        edge_bw = cv2.drawContours(zeros,contours,max_con_arg,(255,255,255),2)
#        edge = cv2.drawContours(im_color,contours,max_con_arg,(255,255,255),2)
#        cv2.imshow('%i'%iimage,cv2.resize(edge_bw,(500,100)))
        boundary = contours[max_con_arg].reshape(contours[max_con_arg].shape[0],2)
        
#        return boundary
        x_min = boundary[:,0].min()
        x_max = boundary[:,0].max()
        
        
        x_interal = x_max-x_min
        cond1 = boundary[:,0]>= x_min+x_interal/6*1
        cond2 = boundary[:,0]<= x_min+x_interal/6*2
        

        cond_and = np.bitwise_and(cond1,cond2)
        

        y_min = int(boundary[cond_and,1].min()*1)
        y_max = int(boundary[cond_and,1].max()*1)
        print(y_min,y_max)
        x_start = x_min-window_black
        x_end = x_min+window_tape
        edge_bw= edge_bw[:,x_start:x_end]
        edge_bw = cv2.cvtColor(edge_bw,cv2.COLOR_BGR2GRAY)
        edge_bw = cv2.resize(edge_bw,(420,2284))
#        cv2.imshow(fname,cv2.resize(edge_bw[:,x_start:x_end],(200,800)))
#        cv2.imshow('alo',cv2.resize(im_color[:,x_start:x_end],(200,800)))

        seg_len = (y_max-y_min)//(nsegments[Q])
        
        segments = []
        
        for iseg in range(0,nsegments[Q]):
#        for iseg in range(1):
            y_start = y_min+iseg*seg_len
            y_end =   y_min+(iseg+1)*seg_len
#            cv2.imwrite('%s_%d.tif'%(fname[:-4],iseg),im_color[y_start:y_end,x_start:x_end])
            
            cond1 = boundary[:,1]>= y_start
            cond2 = boundary[:,1]<= y_end
        

            cond_and = np.bitwise_and(cond1,cond2)
            x_min_sec = boundary[cond_and,0].min()
#            plt.plot(boundary[cond_and,:][:,0],boundary[cond_and,:][:,1])
#            print(x_min_sec)
#            
            x_start = x_min_sec-window_black
            x_end = x_min_sec+window_tape
            
            
            isection = im_color[y_start:y_end,x_start:x_end]
#            isection = cv2.resize(isection,size)
            
            isection = cv2.copyMakeBorder(isection,1,1,1,1,cv2.BORDER_CONSTANT,value=[0,0,0])
            segments.append(cv2.cvtColor(isection,cv2.COLOR_BGR2GRAY))
        img_seg = cv2.vconcat(segments) 
##        plt.figure()
##        plt.imshow(img_seg)
        cv2.imshow('segmented1 %i'%iimage,img_seg)
##                cv2.imwrite("segmented.tif",cv2.vconcat(segments))
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()  
        segments_4p = []
        seg_len = (y_max-y_min)//(4)
        for iseg in range(0,4):
            y_start = y_min+iseg*seg_len
            y_end =   y_min+(iseg+1)*seg_len
            
            
            cond1 = boundary[:,1]>= y_start
            cond2 = boundary[:,1]<= y_end
        

            cond_and = np.bitwise_and(cond1,cond2)
            x_min_sec = boundary[cond_and,0].min()
#            plt.plot(boundary[cond_and,:][:,0],boundary[cond_and,:][:,1])
#            print(x_min_sec)
            
            x_start = x_min_sec-window_black_4p
            x_end = x_min_sec+window_tape_4p
            
#            cv2.imwrite('%s_%d.tif'%(fname[:-4],iseg),im_color[y_start:y_end,x_start:x_end])
            isection = im_color[y_start:y_end,x_start:x_end]
#            isection = cv2.resize(isection,size_4p)
            
            isection = cv2.copyMakeBorder(isection,1,1,1,1,cv2.BORDER_CONSTANT,value=[0,0,0])
            segments_4p.append(cv2.cvtColor(isection,cv2.COLOR_BGR2GRAY))
        img_seg = cv2.vconcat(segments_4p) 
#        plt.figure()
#        plt.imshow(img_seg)
        cv2.imshow('segmented%i'%iimage,img_seg)
#                cv2.imwrite("segmented.tif",cv2.vconcat(segments))
        cv2.waitKey(0)
#        cv2.destroyAllWindows()  
        

        x_interal = x_max-x_min
        cond1 = boundary[:,0]>= x_min+x_interal/6*0
        cond2 = boundary[:,0]<= x_min+x_interal/6*1
        cond_12 = np.bitwise_and(cond1,cond2)
        
#        y_min = int(boundary[cond_and,1].min()*1)
#        y_max = int(boundary[cond_and,1].max()*1)
#        
#        x_start = x_min-window_black
#        x_end = x_min+window_tape
        edge = boundary[cond_12]
        y_min = min(edge[:,1])
        y_max = max(edge[:,1])
        y_interval = y_max - y_min
        
#        argsort = np.argsort(test[:,1])

        npoints = 1000
        data = np.zeros((npoints,2))
        
        for ipoint in range(0,npoints):
            y_start = y_min+ipoint*(y_interval/npoints)
            y_end   = y_min+(ipoint+1)*(y_interval/npoints)
            cond1 = edge[:,1] >= y_start
            cond2 = edge[:,1] <= y_end
            cond_and = np.bitwise_and(cond1,cond2)
            data[ipoint,:] = np.average(edge[cond_and],axis=0)
#        data = boundary[cond_12]
        
#        plt.figure()
#        plt.scatter(data[:,0],data[:,1],s=1)

#        print(edge_bw.shape)
        return data,segments,segments_4p,edge_bw,boundary
        



ls = os.listdir('all')
matches = pd.read_excel('Match_Files.xlsx')
ndata_point = 1372
#rf = h5py.File('TapeMatchData.h5py','w')
#dset_line = rf.create_dataset('line',(1372,1000),dtype=np.float64)
#dset_segme
#rf.close()


c=0
DATA = []
SEGMENTS = []
SEGMENTS_4p = []
EDGE_BW = []
LABELS = []
NAMES = []

for imatch in range(len(matches)):
#    if c >= 2 :
#        continue
#    if imatch <394:
#        continue
    t1 = matches.iloc[imatch]['Tape_1_1']
    t2 = matches.iloc[imatch]['Tape_2_1']
    r1 = matches.iloc[imatch]['Rotation1']
    t3 = matches.iloc[imatch]['Tape_1_2']
    t4 = matches.iloc[imatch]['Tape_2_2']
    r2 = matches.iloc[imatch]['Rotation2']
#    print(imatch)
    fnames = [t1[:-2]+'.tif',t2[:-2]+'.tif',t3[:-2]+'.tif',t4[:-2]+'.tif']
    sides = [t1[-1],t2[-1],t3[-1],t4[-1]]
    rotations = [0,r1,0,r2]
    for ifname in range(4) :
        fname = fnames[ifname]
        if fname == 'LQ_420.tif' or fname == 'LQ_419.tif':
            window_tape = 600
        if os.path.exists('all'+os.sep+'NoString_'+fname):
            fnames[ifname] = 'NoString_'+fname
    DATA.append([])
    SEGMENTS.append([])
    SEGMENTS_4p.append([])
    EDGE_BW.append([])
    LABELS.append(1)
    for iimage in range(4):
        
        data,segments,segments_4p,edge_bw,boundary = get_data(fnames[iimage],sides[iimage],rotations[iimage])
        DATA[-1].append(data)
        SEGMENTS[-1].append(segments)
        SEGMENTS_4p[-1].append(segments_4p)
        EDGE_BW[-1].append(edge_bw)
    c+=1
#    cv2.waitKey(0)
    cv2.destroyAllWindows() 
            
#non_matches = pd.read_excel('Non_Match_Files.xlsx')

#c = 0        
#for imatch in range(len(non_matches)):
##    if c >= 1 :
##        continue
#    if imatch > 972:
#        continue
#    c+=1
#    
#    
#    t1 = non_matches.iloc[imatch]['Tape_1_1']
#    t2 = non_matches.iloc[imatch]['Tape_2_1']
#    r1 = non_matches.iloc[imatch]['Rotation1']
#    t3 = non_matches.iloc[imatch]['Tape_1_2']
#    t4 = non_matches.iloc[imatch]['Tape_2_2']
#    r2 = non_matches.iloc[imatch]['Rotation2']
#    
##    print(imatch)
#    fnames = [t1[:-2]+'.tif',t2[:-2]+'.tif',t3[:-2]+'.tif',t4[:-2]+'.tif']
#    if 'LQ_632.tif' in fnames or 'LQ_772.tif' in fnames or 'HQC_172.tif' in fnames or 'HQC_196.tif' in fnames:
#        continue
#        
#    sides = [t1[-1],t2[-1],t3[-1],t4[-1]]
#    rotations = [0,r1,0,r2]
#    for ifname in range(4) :
#        fname = fnames[ifname]
#        if fname == 'LQ_420.tif' or fname == 'LQ_419.tif':
#            window_tape = 600
#        if os.path.exists('all'+os.sep+'NoString_'+fname):
#            fnames[ifname] = 'NoString_'+fname
#    DATA.append([])
#    SEGMENTS.append([])
#    SEGMENTS_4p.append([])
#    EDGE_BW.append([])
#    LABELS.append(0)
#    for iimage in range(4):
#        # data approach3
#        # segments approach1
#        # edge_bw apparch2
#       data,segments,segments_4p,edge_bw = get_data(fnames[iimage],sides[iimage],rotations[iimage])
#       DATA[-1].append(data)
#       SEGMENTS[-1].append(segments)
#       SEGMENTS_4p[-1].append(segments_4p)
#       EDGE_BW[-1].append(edge_bw)