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


def get_dataloader(config, mode='train', batchsize=64, width=96, height=96):

    train_datas = []
    train_masks = []
    for index in config.training_fold_index:
        tempdata = readCSV(os.path.join(config.csvPath, 'data_fold' + str(index) + '.csv'))
        tempmask = readCSV(os.path.join(config.csvPath, 'mask_fold' + str(index) + '.csv'))

        train_datas += tempdata
        train_masks += tempmask

    test_datas = readCSV(os.path.join(config.csvPath, 'data_fold' + str(config.test_fold_index[0]) + '.csv'))
    test_masks = readCSV(os.path.join(config.csvPath, 'mask_fold' + str(config.test_fold_index[0]) + '.csv'))

    
    if mode=='train':
        # remove features labels
        temp_train_datas = []
        for one in train_datas:
            one_temp = one.split('/')[-1]
            one_list = one_temp.split('_')
            temp_train_datas.append(one_list[0] + '_' + one_list[1] + '_' + one_list[2])
        temp_test_datas = []
        for one in test_datas:
            one_temp = one.split('/')[-1]
            one_list = one_temp.split('_')
            temp_test_datas.append(one_list[0] + '_' + one_list[1] + '_' + one_list[2])

        llower_files = os.listdir(config.llowerPath)
        lower_files = os.listdir(config.lowerPath)
        mid_files = os.listdir(config.midPath)
        upper_files = os.listdir(config.upperPath)
        uupper_files = os.listdir(config.uupperPath)

        # process training data
        temp2_train_datas_red = []

        temp2_train_red = []
        temp2_train_dif = []
        temp2_train_blue = []
        temp2_train_mask = []

        for one_train_data in temp_train_datas:
            imagename = one_train_data + '.png'
            if imagename in llower_files:
                innertemp0 = config.llowerPath + one_train_data + '.png'
                innertemp2 = config.llowerPath + one_train_data + '_red.png'
                innertemp3 = config.llowerPath + one_train_data + '_dif.png'
                innertemp4 = config.llowerPath + one_train_data + '_blue.png'
                innertemp5 = config.rowPath + 'mid_' + one_train_data + '_mask.png'
                temp2_train_datas_red.append(innertemp0)
                temp2_train_red.append(innertemp2)
                temp2_train_dif.append(innertemp3)
                temp2_train_blue.append(innertemp4)
                temp2_train_mask.append(innertemp5)

            if imagename in lower_files:
                innertemp0 = config.lowerPath + one_train_data + '.png'
                innertemp2 = config.lowerPath + one_train_data + '_red.png'
                innertemp3 = config.lowerPath + one_train_data + '_dif.png'
                innertemp4 = config.lowerPath + one_train_data + '_blue.png'
                innertemp5 = config.rowPath + 'mid_' + one_train_data + '_mask.png'
                temp2_train_datas_red.append(innertemp0)
                temp2_train_red.append(innertemp2)
                temp2_train_dif.append(innertemp3)  
                temp2_train_blue.append(innertemp4)
                temp2_train_mask.append(innertemp5)

            if imagename in mid_files:
                innertemp0 = config.midPath + one_train_data + '.png'
                innertemp2 = config.midPath + one_train_data + '_red.png'
                innertemp3 = config.midPath + one_train_data + '_dif.png'
                innertemp4 = config.midPath + one_train_data + '_blue.png'
                innertemp5 = config.rowPath + 'mid_' + one_train_data + '_mask.png'
                temp2_train_datas_red.append(innertemp0)
                temp2_train_red.append(innertemp2)
                temp2_train_dif.append(innertemp3) 
                temp2_train_blue.append(innertemp4)
                temp2_train_mask.append(innertemp5)

            if imagename in upper_files:
                innertemp0 = config.upperPath + one_train_data + '.png'
                innertemp2 = config.upperPath + one_train_data + '_red.png'
                innertemp3 = config.upperPath + one_train_data + '_dif.png'
                innertemp4 = config.upperPath + one_train_data + '_blue.png'
                innertemp5 = config.rowPath + 'mid_' + one_train_data + '_mask.png'
                temp2_train_datas_red.append(innertemp0)
                temp2_train_red.append(innertemp2)
                temp2_train_dif.append(innertemp3) 
                temp2_train_blue.append(innertemp4)
                temp2_train_mask.append(innertemp5)
    
            if imagename in uupper_files:
                innertemp0 = config.uupperPath + one_train_data + '.png'
                innertemp2 = config.uupperPath + one_train_data + '_red.png'
                innertemp3 = config.uupperPath + one_train_data + '_dif.png'
                innertemp4 = config.uupperPath + one_train_data + '_blue.png'
                innertemp5 = config.rowPath + 'mid_' + one_train_data + '_mask.png'
                temp2_train_datas_red.append(innertemp0)
                temp2_train_red.append(innertemp2)
                temp2_train_dif.append(innertemp3)
                temp2_train_blue.append(innertemp4)
                temp2_train_mask.append(innertemp5)

        train_datas_red = temp2_train_datas_red
        train_reds = temp2_train_red
        train_difs = temp2_train_dif
        train_blues = temp2_train_blue
        train_mask = temp2_train_mask

        temp2_test_datas_red = []
        temp2_test_red = []
        temp2_test_dif = []
        temp2_test_blue = []
        temp2_test_mask = []
        for one_test_data in temp_test_datas:
            imagename = one_test_data + '.png'       
            if imagename in mid_files:
                innertemp0 = config.midPath + one_test_data + '.png'           
                innertemp2 = config.midPath + one_test_data + '_red.png'
                innertemp3 = config.midPath + one_test_data + '_dif.png'
                innertemp4 = config.midPath + one_test_data + '_blue.png'
                innertemp5 = config.rowPath + 'mid_' + one_test_data + '_mask.png'
                temp2_test_datas_red.append(innertemp0)
                temp2_test_red.append(innertemp2)
                temp2_test_dif.append(innertemp3) 
                temp2_test_blue.append(innertemp4)
                temp2_test_mask.append(innertemp5)

        test_datas_red = temp2_test_datas_red
        test_reds = temp2_test_red    
        test_blues = temp2_test_blue
        test_difs = temp2_test_dif
        test_mask = temp2_test_mask

        print('***********')
        print('the length of train data: ', len(train_datas))
        print('the length of test data: ', len(test_datas))
        print('-----------')
        dataloader = loader(Dataset(train_datas_red, train_reds, train_blues, train_difs, train_mask,  width=width, height=height), batchsize)
        dataloader_val = loader(Dataset(test_datas_red, test_reds, test_blues, test_difs, test_mask, width=width, height=height), batchsize)
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

        dataloader = loader(RowDataset(temp2_train_datas, temp2_train_masks, width=width, height=height), batchsize)
        dataloader_val = loader(RowDataset(temp2_test_datas, temp2_test_masks, width=width, height=height), batchsize)

        # for batch_idx, (inputs, targets) in enumerate(dataloader):
        #     inputsfolder = inputs.chunk(inputs.shape[0], dim=0)
        #     xx = inputsfolder[0]
        #     print(xx.shape)
        #     xx = xx.squeeze()
        #     xx = np.transpose(xx.cpu().detach().numpy(), (1, 2, 0))
        #     print(xx.shape)
        #     cv2.imwrite('./test_input.jpg', xx)

        return dataloader, dataloader_val
        
