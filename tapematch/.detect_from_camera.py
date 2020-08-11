# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:51:21 2020

@author: ptava
"""

import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while True :
    _,frame = cap.read()
    bluured_frame = cv2.GaussianBlur(frame,(5,5),0)
    hsv = cv2.cvtColor(bluured_frame,cv2.COLOR_BGR2HSV )
     
    lower_blue = np.array([30,150,150])
    upper_blue = np.array([255,255,180])
    
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    
#    res = cv2.bitwise_and(frame,frame,mask=mask)
#    
#    kernel = np.ones((15,15),np.float32)/255
#    smoothed = cv2.filter2D(res,-1,kernel)
#    blur = cv2.GaussianBlur(res,(15,15),0)
#    median = cv2.medianBlur(res,15)
    
    
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    cv2.drawContours(frame,contours,-1,(0,255,0),3)
    
    
    
    cv2.imshow("Mask",mask)
    cv2.imshow("HSV",hsv)
    cv2.imshow("Frame",frame)
#    cv2.imshow("res",res) 
#    cv2.imshow("blur",blur) 
#    cv2.imshow("median",median) 
    
    key = cv2.waitKeyEx(1)
    if key == 27 :
       break
cap.release()
cv2.destroyAllWindows()  