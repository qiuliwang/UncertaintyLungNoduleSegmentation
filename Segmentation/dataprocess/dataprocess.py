import random
from .segdataloader import *
from .utils import *
import csv
import glob 
import cv2

fold = 1

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line[0])
    return lines


def get_dataloader(config, mode='train', batchsize=64, width=512, height=512):

    train_datas = readCSV(os.path.join(config.csvPath, '512List/training_split_labeled.csv'))
    train_masks = readCSV(os.path.join(config.csvPath, '512List/training_split_labeled.csv'))

    test_datas = readCSV(os.path.join(config.csvPath, '512List/testing_split_labeled.csv'))
    test_masks = readCSV(os.path.join(config.csvPath, '512List/testing_split_labeled.csv'))

    
    if mode=='train':
        # remove features labels
        temp_train_datas = []
        for one in train_datas:
            one_temp = one
            temp_train_datas.append(one_temp)

        temp_test_datas = []
        for one in test_datas:
            one_temp = one 
            temp_test_datas.append(one_temp)

        temp2_train_data = []
        temp2_train_mask = []

        for one_train_data in temp_train_datas:
            image_data = config.image_path + one_train_data + '.jpeg'
            mask_data = config.image_path + one_train_data + '_mask.jpeg'
            temp2_train_data.append(image_data)
            temp2_train_mask.append(mask_data)

        temp2_test_data = []
        temp2_test_mask = []

        for one_test_data in temp_test_datas:
            image_data = config.image_path + one_test_data + '.jpeg'
            mask_data = config.image_path + one_test_data + '_mask.jpeg'
            temp2_test_data.append(image_data)
            temp2_test_mask.append(mask_data)

        print('***********')
        print('the length of train data: ', len(temp2_train_data))
        print('the length of test data: ', len(temp2_test_data))
        print('-----------')
        dataloader = loader(Dataset(temp2_train_data, temp2_train_mask, width = width, height = height), batchsize)
        dataloader_val = loader(Dataset(temp2_test_data, temp2_test_mask, width = width, height = height), batchsize)

        return dataloader, dataloader_val