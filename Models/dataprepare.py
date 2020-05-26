'''
Created by Wang Qiuli, Li Zhihuan
2019/4/8

wangqiuli@cqu.edu.cn
'''

import os
import csvTools
import pydicom
import numpy as np
import tensorflow as tf
import math
import scipy.misc
import operator
cmpfun = operator.attrgetter('InstanceNumber')
from PIL import Image
from random import choice
import cv2

def truncate_hu(image_array, max, min):
    image = image_array.copy()
    image[image > max] = max
    image[image < min] = min
    image = normalazation(image)
    return image
def getThreeChannel(pixhu):
    lungwindow = truncate_hu(pixhu, 400, -1000)
    highattenuation = truncate_hu(pixhu, 240, -160)
    lowattenuation = truncate_hu(pixhu, -950, -1400)
    pngfile = [lowattenuation, lungwindow, highattenuation]
    pngfile = np.array(pngfile).transpose(1,2,0)
    return  pngfile    

def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = image_array.mean()
    image_array = image_array-avg
    return image_array   # a bug here, a array must be returned,directly appling function did't work

def get_pixels_hu(ds):
    image = ds.pixel_array
    image = np.array(image , dtype = np.float32)
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    image = image * slope
    image += intercept
    return image
    
def gray2rgb(im):
    im=im[:,:,np.newaxis]
    im0=im1=im
    
    im=np.concatenate((im0,im1,im),axis=2)
    return im

class Data(object):
    def __init__(self):
	    self.count = 0

    def getOnePatient(self, patientName, transsign = False):
        if 'nor' in patientName[1]:
            datapath = 'path/for/normalpatient/'
        else:
            datapath = 'path/for/pneumonia/'

        dcmfiles = os.listdir(datapath + patientName[0])
        dcmfiles.sort()
        slices = [pydicom.dcmread(os.path.join(datapath, patientName[0], s)) for s in dcmfiles]
        slices.sort(key = cmpfun)
        slicethickness = slices[0].data_element('SliceThickness').value
        dcmkeep = []
        keeprate = 10 / slicethickness
        keeprate = int(math.floor(keeprate))
        if keeprate < 1:
            keeprate = 1

        tempsign = 0
        for onedcm in slices:
            if tempsign % keeprate == 0:
                dcmkeep.append(onedcm)
            tempsign += 1
        if len(dcmkeep) > 32:
            
            dcmkeep = dcmkeep[:32]
        if len(dcmkeep) < 32:
            temp = []
            for i in range(0, 32 - len(dcmkeep)):
                temp.append(dcmkeep[0])
            dcmkeep = temp + dcmkeep

        indexlist = [0,1,2,3]
        pixels = []
        mid = dcmkeep[len(dcmkeep) // 2]
        if angle > 30 or angle < -30:
            angle = 0
        sign = 1
        for temp in dcmkeep:
            temp = get_pixels_hu(temp)
            temp=getThreeChannel(temp)
            temp = cv2.medianBlur(temp, 3)
            sign += 1
            pixels.append(temp)
            

        pixels = np.array(pixels, dtype=np.float)
        if len(pixels.shape)<4:
            pixels = np.expand_dims(pixels, -1)        

        return pixels

    def getThickness(self):
        return self.slicethicknesscount

