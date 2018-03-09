#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 20:21:22 2018

@author: frangarcia
"""
import glob as g
import cv2
import numpy as np


def prepare_all_data():
    pathsTrainPistol=g.glob("Train/Pistol/*[!*.ini]")
    pathsTrainSmartphone=g.glob("Train/Smartphone/*[!*.ini]")
    pathsTest=sorted(g.glob("Test/*[!*.ini]"),key=lambda name: int(name[8:-4]))
    
    pistols=[]
    smartphones=[]
    test=[]
    w=128
    h=128
    nPistols=len(pathsTrainPistol)
    nSmartphones=len(pathsTrainSmartphone)
    totalRows=nPistols + nSmartphones
    labels = np.zeros(shape=(totalRows,2), dtype=int) #o: pistol; 1:smartphone
    
    
    for i,path in enumerate(pathsTrainPistol):
        img=cv2.imread(path)
        img=cv2.resize(img,(w,h))
        pistols.append(img)
        labels[i,0]=1
       
    for i,path in enumerate(pathsTrainSmartphone):
        img=cv2.imread(path)
        img=cv2.resize(img,(w,h))
        smartphones.append(img)
        labels[i+nPistols,1]=1
        
    for path in pathsTest:
        img=cv2.imread(path)
        img=cv2.resize(img,(w,h))
        test.append(img)
        
    data=np.append(pistols, smartphones, axis=0)
    test=np.asarray(test)
    
    return (data, labels, test)