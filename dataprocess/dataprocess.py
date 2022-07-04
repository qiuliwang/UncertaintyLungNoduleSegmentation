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

    train_datas = readCSV(os.path.join(config.csvPath, 'train_csv.csv'))
    train_masks = readCSV(os.path.join(config.csvPath, 'train_csv.csv'))

    test_datas = readCSV(os.path.join(config.csvPath, 'test_csv.csv'))
    test_masks = readCSV(os.path.join(config.csvPath, 'test_csv.csv'))

    
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
        temp2_train_core = []
        temp2_train_blood = []

        for one_train_data in temp_train_datas:
            image_data = config.image_path + one_train_data + '_grey.png'
            mask_data = config.image_path + one_train_data + '_mask.png'
            core_data = config.image_path + one_train_data + '_core.png'
            blood_data = config.image_path + one_train_data + '_blood.png'
            
            temp2_train_data.append(image_data)
            temp2_train_mask.append(mask_data)
            temp2_train_core.append(core_data)
            temp2_train_blood.append(blood_data)


        temp2_test_data = []
        temp2_test_mask = []
        temp2_test_core = []
        temp2_test_blood = []

        for one_test_data in temp_test_datas:
            image_data = config.image_path + one_test_data + '_grey.png'
            mask_data = config.image_path + one_test_data + '_mask.png'
            core_data = config.image_path + one_train_data + '_core.png'
            blood_data = config.image_path + one_train_data + '_blood.png'

            temp2_test_data.append(image_data)
            temp2_test_mask.append(mask_data)
            temp2_test_core.append(core_data)
            temp2_test_blood.append(blood_data)

        print('***********')
        print('the length of train data: ', len(temp2_train_data))
        print('the length of test data: ', len(temp2_test_data))
        print('-----------')
        dataloader = loader(Dataset(temp2_train_data, temp2_train_core, temp2_train_blood, temp2_train_mask, width = width, height = height), batchsize)
        dataloader_val = loader(Dataset(temp2_test_data, temp2_test_core, temp2_test_blood, temp2_test_mask, width = width, height = height), batchsize)
        return dataloader, dataloader_val

    if mode=='row':
        # remove features labels
        temp_train_datas = []
        for one in train_datas:
            one_temp = one.split('/')[-1]
            one_list = one_temp.split('_')
            temp_train_datas.append('mid_' + one_list[0] + '_' + one_list[1] + '_' + one_list[2])
        temp_test_datas = []
        for one in test_datas:
            one_temp = one.split('/')[-1]
            one_list = one_temp.split('_')
            temp_test_datas.append('mid_' + one_list[0] + '_' + one_list[1] + '_' + one_list[2])
        temp2_train_datas = []
        temp2_train_masks = []
        temp2_test_datas = []
        temp2_test_masks = []
        row_files = os.listdir(config.rowPath)
        for one_train_data in temp_train_datas:
            imagename = one_train_data + '.png'
            if imagename in row_files:
                innertemp0 = config.rowPath + one_train_data + '.png'
                innertemp1 = config.rowPath + one_train_data + '_mask.png'
                temp2_train_datas.append(innertemp0)
                temp2_train_masks.append(innertemp1)
        for one_test_data in temp_test_datas:
            imagename = one_test_data + '.png'
            if imagename in row_files:
                innertemp0 = config.rowPath + one_test_data + '.png'
                innertemp1 = config.rowPath + one_test_data + '_mask.png'
                temp2_test_datas.append(innertemp0)
                temp2_test_masks.append(innertemp1)

        dataloader = loader(RowDataset(temp2_train_datas, temp2_train_core, temp2_train_blood, temp2_train_masks, width=width, height=height), batchsize)
        dataloader_val = loader(RowDataset(temp2_test_datas, temp2_test_core, temp2_test_blood, temp2_test_masks, width=width, height=height), batchsize)

        return dataloader, dataloader_val
        
