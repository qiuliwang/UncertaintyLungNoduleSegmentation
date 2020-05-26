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

sample = []
sample_mask = []
for one1 in data:
    if 'LIDC-IDRI-0055_3_3000581' in one1:
        sample.append(one1)
for one2 in mask:
    if 'LIDC-IDRI-0055_3_3000581' in one2:
        sample_mask.append(one2)

print(sample)
print(sample_mask)

data_list = loadList(data_path, sample)
mask_list = loadList(mask_path, sample_mask)

for onedata in data_list:
    io.imsave('onedata.png', onedata)

    hist_value = hist(onedata)

    superpixel_value = super_pixel(onedata)
    io.imsave('superpixel_value.png', superpixel_value)

    std = standard_deviation(onedata)

    edge = edge_detection(onedata)
    io.imsave('edge.png', edge)

    gabor_value = gabor(onedata)
    io.imsave('gabor_value.png', gabor_value)

    void = threshold_void(onedata)
    io.imsave('void.png', void)

    grey_level = toGrey(onedata)
    io.imsave('grey_level.png', grey_level)
    
    # OTSU_value = OTSU(onedata)
