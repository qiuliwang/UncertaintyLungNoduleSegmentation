'''
Created by Wang Qiuli

2020/5/23
'''

import cv2
import numpy as np 
import matplotlib.pyplot as plt
import slic
import rgb_gr
from PIL import Image
from skimage import io, color


def truncate_hu(image_array, max = 400, min = -900):
    image = image_array.copy()
    image[image > max] = max
    image[image < min] = min
    return image

def hist(img):
    '''
    return histgram values
    1 * 128
    '''
    img = truncate_hu(img)
    hist = cv2.calcHist([img],[0],None,[128],[-900,400])
    # print(hist.shape)
    # plt.subplot(121)
    # plt.imshow(img,'gray')
    # plt.xticks([])
    # plt.yticks([])
    # plt.title("Original")
    # plt.subplot(122)
    # plt.hist(img.ravel(),128,[-900,400])
    # plt.show()    
    return hist

def gray2rgb(rgb,imggray):
   R = rgb[:,:,0]
   G = rgb[:,:,1]
   B = ((imggray) - 0.299 * R - 0.587 * G) / 0.114
 
   grayRgb = np.zeros((rgb.shape))
   grayRgb[:, :, 2] = B
   grayRgb[:, :, 0] = R
   grayRgb[:, :, 1] = G
 
   return grayRgb

def super_pixel(img):
    '''
    return super_pixel images
    img w * h
    '''
    img = truncate_hu(img)
    # io.imsave('ori.png', img)
    img = np.expand_dims(img, 2)
    # # print(img.shape)
    rgb = np.concatenate((img, img, img), 2)
    # io.imsave('ori2.png', rgb)
    obj = slic.SLICProcessor(rgb, 80, 10)
    res = obj.iterate_10times()
    return res

def standard_deviation(img):
    hist_value = hist(img)
    print(hist_value.shape)
    # return np.std(hist_value)

