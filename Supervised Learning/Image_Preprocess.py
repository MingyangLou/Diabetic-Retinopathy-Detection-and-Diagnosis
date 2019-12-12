#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:10:13 2019

@author: l.kate
"""

# Load Packages
import json #read json file to load the label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
import os
from glob import glob
import re
import sys
import cv2
from subprocess import check_output

def scaleRadius(img,scale):
    x = img [img.shape[0]//2,:,:].sum(1)
    r = (x>x.mean()//10).sum()/2
    s = scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)

def img_preprocess(Dir,scale):
    #global ImageNameDataHash
    # loop over the input images
    images = os.listdir(Dir)
    print("Number of files in " + Dir + " is " + str(len(images)))
    for imageFileName in images:
        # load the image, pre-process it, and store it in the data list
        imageFullPath = os.path.join(Dir, imageFileName)
        a=cv2.imread(imageFullPath)
        try :
            #scale img to a given radius
            a=scaleRadius(a,scale)
            #subtract local mean color
            a=cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale//30),-4,128)
            #remove outer 10%
            b=np.zeros(a.shape)
            cv2.circle(b,(a.shape[1]//2,a.shape[0]//2),int(scale*0.9),(1,1,1),-1,8,0)
            a=a*b + 128 * (1 - b)
            
        except:
            print(imageFullPath)
        a=cv2.resize(a,(256,256))
        cv2.imwrite(os.path.join('.',str(scale)+'_preprocessed',imageFileName),a)
        
scale = 500
img_preprocess('./images',scale)