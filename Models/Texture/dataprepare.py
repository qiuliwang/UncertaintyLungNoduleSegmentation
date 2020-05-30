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
import rotate

class Data(object):
    def __init__(self):
	    self.count = 0        

    def truncate_hu(self, image_array, max = 400, min = -900):
        image = image_array
        image[image > max] = max
        image[image < min] = min
        return image

    def normalization(self, image_array):
        image_array = image_array + 900

        max = 1300.0
        min = 0
        image_array = (image_array-min)/(max-min)
        # avg = image_array.mean()
        # image_array = image_array-avg
        image_array = image_array
        # image_array.dtype = 'int16'
        pngfile = np.array([image_array, image_array, image_array])
        pngfile = pngfile.transpose(1, 2, 0)
        return pngfile  

    def angle_transpose(self, array,degree):
        '''
        Modified by Wang Qiuli
        2020/5.28
        @param file : a npy file which store all information of one cubic
        @param degree: how many degree will the image be transposed,90,180,270 are OK
        '''
        
        newarr = np.rot90(array, degree)
        # array = np.load(file)
        # array = array.transpose(2, 1, 0)  # from x,y,z to z,y,x

        # newarr = np.zeros(array.shape,dtype=np.float32)
        # for depth in range(array.shape[0]):
        #     jpg = array[depth]
        #     jpg.reshape((jpg.shape[0],jpg.shape[1],1))
        #     img = Image.fromarray(jpg)
        #     #img.show()
        #     out = img.rotate(degree)
        #     newarr[depth,:,:] = np.array(out).reshape(array.shape[1],-1)[:,:]
        # newarr = newarr.transpose(2,1,0)
        return newarr

    def get_images_and_labels(self, data_root_dir, isTrain = False):
        '''
        Modified by WangQL
        2020/5/27
        '''
        data = data_root_dir

        all_image_path = []
        all_image_label = []

        data_path = '/home/wangqiuli/raid/senet_data/ori_hu/'
        npy_files = os.listdir(data_path)

        for one_data in data:
            id = one_data[0]
            if 'low' in one_data[1]:
                label = [0, 1]
            else:
                label = [1, 0]

            # print(id)
            npy = np.load(os.path.join(data_path, one_data[0] + '.npy'))
            npy = self.truncate_hu(npy)
            nor_npy = self.normalization(npy)
            # print(nor_npy)
            if isTrain:
                if 'low' in one_data[1]:            
                    all_image_path.append(nor_npy)
                    all_image_label.append(label)
                    two = self.angle_transpose(nor_npy, 2)
                    all_image_path.append(two)
                    all_image_label.append(label)

                else:
                    all_image_path.append(nor_npy)
                    all_image_label.append(label)

                    # one = self.angle_transpose(nor_npy, 1)
                    # all_image_path.append(one)
                    # all_image_label.append(label)
                    
                    # two = self.angle_transpose(nor_npy, 2)
                    # all_image_path.append(two)
                    # all_image_label.append(label)

                    # three = self.angle_transpose(nor_npy, 3)
                    # all_image_path.append(three)
                    # all_image_label.append(label)
            else:
                all_image_path.append(nor_npy)
                all_image_label.append(label)

        return all_image_path, all_image_label

    def get_batch_files(self, batch, stat):
        '''
        stat has train, valid, test
        '''
        print(batch)
        slices = []
        labels = []
        return slices, labels

    def get_all_data(self, train, valid, test):
        print('Preparing all data.')
        traindata, trainlabel = self.get_images_and_labels(train, isTrain = True)
        validdata, validlabel = self.get_images_and_labels(valid)
        testdata, testlabel = self.get_images_and_labels(test)
        return traindata, trainlabel, validdata, validlabel, testdata, testlabel