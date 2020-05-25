'''
Created by Wang Qiuli

2020/5/23
'''

import numpy as np 
import os

data_path = ''

def loadList(nodule_list):
    nplist = []
    for one_nodule in nodule_list:
        nodule = np.load(os.path.join(data_path, one_nodule))
        nplist.append(nodule)
    
    return nplist

