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


nsegments = {"LQ": 32, "MQ" : 37,"HQ": 55}
window_tape = 200
window_black = 20
rf = open('li.txt','r')
done_list = rf.read()
rf.close()

ls = os.listdir('all')

matches = pd.read_excel('Match_Files.xlsx')

c = 0
for imatch in range(len(matches)):
    t1 = matches.iloc[imatch]['Tape_1_1']
    t2 = matches.iloc[imatch]['Tape_2_1']
    t3 = matches.iloc[imatch]['Tape_1_2']
    t4 = matches.iloc[imatch]['Tape_2_2']
    if not t1[:-2]+'.tif' in ls :
        print(t1)
    if not t2[:-2]+'.tif' in ls :
        print(t2)
    if not t3[:-2]+'.tif' in ls :
        print(t3)
    if not t4[:-2]+'.tif' in ls :
        print(t4)

non_matches = pd.read_excel('non_matches.xlsx')
for inonmatch in range(1,len(non_matches)):
    t1 = non_matches.iloc[inonmatch]['file1']
    t2 = non_matches.iloc[inonmatch]['file2']
    t3 = non_matches.iloc[inonmatch]['file3']
    t4 = non_matches.iloc[inonmatch]['file4']
    print(t1,t2,t3,t4)
    if not t1[:-2]+'.tif' in ls :
        print(t1)
    if not t2[:-2]+'.tif' in ls :
        print(t2)
    if not t3[:-2]+'.tif' in ls :
        print(t3)
    if not t4[:-2]+'.tif' in ls :
        print(t4)

# print(non_matches)


#for idir in ls :
#    if not os.path.isdir(idir) or idir in ['MQSC']:
#        continue
#    
#    ls2 = os.listdir(idir)
#    for ifile in ls2 :
#        path = "{}{}{}".format(idir,os.sep,ifile) 
#        if path in done_list :
#            continue
#        
#        print(path)
#
#        if '.tif' in ifile and not 'original' in ifile:
#            
##            continue
#            
#            #fname = 'MQHT_223.tif'
#            fname = path
#            #fname = 'MQHT_509.tif'
#            #fname = 'MQHT_506.tif'
#            if 'LQ' in fname :
#                Q = 'LQ'
#            elif 'MQ' in fname :
#                Q = "MQ"
#            else : 
#                Q = "HQ"
#                    
##            if c > 0 :
##                continue
##            c+=1            
#            
#            
#            
#                
#            imgray = cv2.imread(fname,0)
#            
#            w,h = imgray.shape
#            m = h//2
#            img_L = imgray[:, 0:m]
#            img_R = imgray[:,m:h]
#            img_R = cv2.flip(img_R,1)
##            for image in [img_L,img_R]:
#            for image in [img_L]:
#
##                blured_imgray = cv2.GaussianBlur(image,(5,5),0)
#                blured_imgray = image 
#                imgray_resize = cv2.resize(blured_imgray,(500,500))
#                im_color = cv2.cvtColor(imgray_resize,cv2.COLOR_GRAY2BGR )
#                mask = cv2.inRange(imgray_resize,60,255)
#                
#                res = cv2.bitwise_and(imgray_resize,imgray_resize,mask=mask)
#                
#                
##                kernel = np.ones((15,15),np.float32)/255
##                smoothed = cv2.filter2D(res,-1,kernel)
#                
#                
#                contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#                max_con = 0
#                max_con_arg = 0
#                for ic in range(len(contours)):
#                    if cv2.contourArea(contours[ic]) > max_con :
#                        max_con = cv2.contourArea(contours[ic])
#                        max_con_arg = ic
#                cv2.drawContours(im_color,contours,max_con_arg,(0,0,255),2)
#                cv2.imshow("{}".format(ifile),im_color)
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()  
#                
#                #cv2.imwrite('boundary.tif',im_color)
#                boundary = contours[max_con_arg].reshape(contours[max_con_arg].shape[0],2)
#                plt.plot(boundary[:,0],boundary[:,1])
#                
#                y_min = int(boundary[:,1].min()*1)
#                y_max = int(boundary[:,1].max()*1)
#                
#                x_min = boundary[:,0].min()
#                x_max = boundary[:,0].max()
#                
#                cond1 = boundary[:,1]> y_max*0.95
#                cond2 = boundary[:,1]< y_max*1.05
#                
#                cond3 = boundary[:,0]> x_min*1.1
#                cond4 = boundary[:,0]< x_max*0.9
##                cond1 = boundary[:,0]> x_max*0.95
##                cond2 = boundary[:,0]< x_max*1.05
#                
#                cond_12 = np.bitwise_and(cond1,cond2)
#                cond_34 = np.bitwise_and(cond3,cond4)
#                cond_and = np.bitwise_and(cond_12,cond_34)
##                cond_and = np.bitwise_and(cond1,cond2)
#                start = 0
#                end = len(boundary[cond_and][:,0])-0
#                
#                # This part is to rotate the images
#                npoints = 100
#                plt.plot(boundary[cond_and][start:start+npoints,0],boundary[cond_and][start:start+npoints,1])
#                y2 = np.average(boundary[cond_and][start:start+npoints,1])
#                y1 = np.average(boundary[cond_and][end-npoints:end,1])
#                
#                x2 = np.average(boundary[cond_and][start:start+npoints,0])
#                x1 = np.average(boundary[cond_and][end-npoints:end,0])
#                
#                angle = np.arctan((y2-y1)/(x2-x1))
##                angle=0
#                rotated = imutils.rotate(im_color,np.rad2deg(angle))
#                
#                ##################################################################
#                # do the exact same thing on the rotated image
#                im_color = rotated
#                #im_color = cv2.cvtColor(imgray_resize,cv2.COLOR_GRAY2BGR )
#                imgray_resize = cv2.cvtColor(im_color,cv2.COLOR_BGR2GRAY)
#                
#                mask = cv2.inRange(imgray_resize,60,255)
##                cv2.imshow("mask",mask)
#                
#                res = cv2.bitwise_and(imgray_resize,imgray_resize,mask=mask)
#                #cv2.imshow("res",res)
#                
#                
#                
#                contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#                max_con = 0
#                max_con_arg = 0
#                for ic in range(len(contours)):
#                    if cv2.contourArea(contours[ic]) > max_con :
#                        max_con = cv2.contourArea(contours[ic])
#                        max_con_arg = ic
#                zeros = np.zeros_like(im_color)        
#                cv2.drawContours(zeros,contours,max_con_arg,(0,0,255),2)
#                boundary = contours[max_con_arg].reshape(contours[max_con_arg].shape[0],2)
#                plt.plot(boundary[:,0],boundary[:,1],alpha=0.1)
#                
#                x_min = boundary[:,0].min()
#                x_max = boundary[:,0].max()
#                
#                cond1 = boundary[:,0]> x_min*0.90
#                cond2 = boundary[:,0]< x_min*1.1
#                cond_and = np.bitwise_and(cond1,cond2)
#                
#                #int(x_min*0.1):int(x_min*1.2)
#                # if L use x_min, if R use x_
#                y_min = int(boundary[cond_and,1].min()*1)
#                y_max = int(boundary[cond_and,1].max()*1)
#                
#
##                cv2.imshow('IM_COLOR',im_color)
#                x_start = x_min-window_black
#                if x_start < 0 : 
#                    x_start = 0
#                
#                x_end = x_min+window_tape
#                if x_end > im_color.shape[1]-1 :
#                    x_end = im_color.shape[1]-1
#                cv2.imwrite('Rotated.tif',zeros[:,x_start:x_end])
#                
#                
#                seg_len = (y_max-y_min)//(nsegments[Q])
#                
#                segments = []
#                
#                for iseg in range(0,nsegments[Q]):
#                    y_start = y_min+iseg*seg_len
#                    y_end =   y_min+(iseg+1)*seg_len
#                #    y_start = y_max+(iseg-1)*seg_len
#                #    y_end = y_max+iseg*seg_len
#                #    cond1 = boundary[:,1]> y_start
#                #    cond2 = boundary[:,1]< y_end
#                #    cond_and = np.bitwise_and(cond1,cond2)
#                #    x_min_win = boundary[cond_and][:,0].min()
#                    
#                
#                    if y_end > im_color.shape[0]-1:
#                        y_end = im_color.shape[0]-1
##                    cv2.imwrite('%s_%d.tif'%(fname[:-4],iseg),im_color[y_start:y_end,x_start:x_end])
#                    isection = cv2.copyMakeBorder(im_color[y_start:y_end,x_start:x_end],1,1,1,1,cv2.BORDER_CONSTANT,value=[0,0,0])
#                    segments.append(isection)
#                
#                img_seg = cv2.vconcat(segments)
#                #plt.imshow(img_seg)
##                cv2.imshow('segmented',cv2.vconcat(segments))
##                cv2.imwrite("segmented.tif",cv2.vconcat(segments))
##                cv2.waitKey(0)
##                cv2.destroyAllWindows()
#                #    
#                plt.clf()
