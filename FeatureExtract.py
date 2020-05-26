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
from skimage import data, filters


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
    std = np.std(hist_value)
    # print(std)

    return std

def edge_detection(img):
    '''
    edge detection
    '''
    img = truncate_hu(img)
    # io.imsave('ori.png', img)
    # img = np.expand_dims(img, 2)
    # # # print(img.shape)
    # rgb = np.concatenate((img, img, img), 2)

    # gray= cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)


    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    z = cv2.Sobel(img, cv2.CV_16S, 1, 1)

    return x
    # io.imsave('canny1.png', x)
    # io.imsave('canny2.png', y)
    # io.imsave('canny3.png', z)


def gabor(img):
    filt_real, filt_imag = filters.gabor(img,frequency=0.6)   
    # io.imsave('filt_imag.png', filt_imag)
    return filt_imag


# def glcm(img, d_x, d_y, gray_level=16):
#     '''Gray Level Co-occurrence Matrix'''
#     img = np.expand_dims(img, 2)
#     # # print(img.shape)
#     rgb = np.concatenate((img, img, img), 2)

#     arr= cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
#     print(arr)

#     max_gray = arr.max()
#     height, width = arr.shape
#     arr = arr.astype(np.float64)  
#     arr = arr * (gray_level - 1) // max_gray 
#     ret = np.zeros([gray_level, gray_level])
#     for j in range(height -  abs(d_y)):
#         for i in range(width - abs(d_x)):  
#             rows = arr[j][i].astype(int)
#             cols = arr[j + d_y][i + d_x].astype(int)
#             ret[rows][cols] += 1
#     if d_x >= d_y:
#         ret = ret / float(height * (width - 1))  
#     else:
#         ret = ret / float((height - 1) * (width - 1))  
#     return ret
