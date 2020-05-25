'''
Created by Wang Qiuli

2020/5/25
'''

import os 
from DataLoader import *
from FeatureExtract import *
from skimage import io, color


data_path = 'D:/Data/nodules/ori_hu'
mask_path = 'D:/Data/nodules/ori_masks'

data = os.listdir(data_path)
mask = os.listdir(mask_path)
print('number of data: ', len(data))
print('number of mask: ', len(mask))

sample = data[0:10]
sample_mask = mask[0:10]
print(sample)
print(sample_mask)

data_list = loadList(data_path, sample)
mask_list = loadList(mask_path, sample_mask)

for onedata in data_list:
    # hist_value = hist(onedata)
    # superpixel_value = super_pixel(onedata)
    std = super_pixel(onedata)
    print(std.shape)