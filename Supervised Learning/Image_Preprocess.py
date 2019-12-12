#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:10:13 2019

@author: l.kate
"""

# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from skimage.io import imread
from glob import glob
import sys
import cv2
from subprocess import check_output
from PIL import Image, ImageEnhance
import random
from scipy import ndarray
import skimage as sk
from skimage import transform, util

os.chdir('/Users/l.kate/Downloads/FinTech_DR')

def scaleRadius(img,scale):
    x = img [img.shape[0]//2,:,:].sum(1)
    r = (x>x.mean()//10).sum()/2
    s = scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)

def img_preprocess(Dir,scale):
    '''
    Params:
    Dir: path to images
    scale: desired pixels
    
    img_preprocess
        * Performs rescale
        * Map RBG to 50% gray
        * Clip the image to 90% size
    Return:
        write the new image to the disk
    '''
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
        
def img_enhancement(Dir):
    '''
    Params:
    Dir: path to images
    
    img_enhancement
        * Performs image enhancement
    Return:
        write the new image to the disk
    '''
    images = os.listdir(Dir)
    print("Number of files in " + Dir + " is " + str(len(images)))
    for imageFileName in images:
        imageFullPath = os.path.join(Dir, imageFileName)
        a = Image.open(imageFullPath)
        try:
            # Enhancement
            a = ImageEnhance.Contrast(a).enhance(2.0)
            enhanced_image = np.asarray(a)
            b,g,r  = cv2.split(enhanced_image)
            a = cv2.merge([b, g, r])
        except:
            print(imageFullPath)
        cv2.imwrite(os.path.join('.',str(scale)+'_preprocessed_enhanced',imageFileName),a)


def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

def img_augmentation(Dir):
    '''
    Params:
    Dir: path to images
    
    img_augmentation
        * Performs random augmentaions to images with some of following adjustments
            - Random rotation from -25 to 25 degrees
            - Add random noise
            - Horizontal flip
    Return:
        write the new image to the disk
    '''
    images = os.listdir(Dir)
    
    # dictionary of the transformations functions we defined earlier
    available_transformations = {
        'rotate': random_rotation,
        'noise': random_noise,
        'horizontal_flip': horizontal_flip
    }
     
    for imageFileName in images:
        imageFullPath = os.path.join(Dir,imageFileName)
        # read image as an two dimensional array of pixels
        image_to_transform = sk.io.imread(imageFullPath)
    
    
        # random num of transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
    
        num_transformations = 0
        transformed_image = None
        while num_transformations <= num_transformations_to_apply:
            # choose a random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1
        
        # write image to the disk
        sk.io.imsave(os.path.join('.',str(scale)+'_enhanced_augmentated',imageFileName), (transformed_image*255).astype(np.uint8))  
        
scale = 500
img_preprocess('images',scale)
img_enhancement('500_preprocessed')
img_augmentation('500_preprocessed_enhanced')

