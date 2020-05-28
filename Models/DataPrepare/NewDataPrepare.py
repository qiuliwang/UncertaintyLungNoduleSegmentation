'''
Created by WangQL
2020/5/27

'''

import csvTools
import os

labelCSV = csvTools.readCSV('D:/Data/nodules/malignancy.csv')
print('number of samples: ', len(labelCSV))

data_path = 'D:/Data/nodules/ori_hu/'
npy_data = os.listdir(os.path.join(data_path))
print('number of npy: ', len(npy_data))

data_dic_low = {}
data_dic_high = {}
data_dic_mid = {}

for one_npy in npy_data:
    temp_name = one_npy
    patient_id = temp_name[:temp_name.find('_')]
    patient_id = int(patient_id[len(patient_id) - 4:])
    # print(patient_id)

    temp_name = temp_name[temp_name.find('_') + 1:]
    nodule_id = temp_name[:temp_name.find('_')]
    # print(nodule_id)
    
    temp_name = temp_name[temp_name.find('_') + 1:]
    scan_id = temp_name[:temp_name.find('.')]
    # print(scan_id)

    for onelabel in labelCSV:
        if int(patient_id) == int(onelabel[1]):
            if str(scan_id) in str(onelabel[2]) or str(onelabel[2]) in str(scan_id):
                if int(nodule_id) == int(onelabel[3]):

                    '''
                    <subtlety>5</subtlety>
                    <internalStructure>1</internalStructure>
                    <calcification>6</calcification>
                    <sphericity>3</sphericity>
                    <margin>3</margin>
                    <lobulation>3</lobulation>
                    <spiculation>4</spiculation>
                    <texture>5</texture>
                    <malignancy>5</malignancy>
                    '''    
                    malignancy = float(onelabel[29])
                    if malignancy >= 3.5:
                        data_dic_high[one_npy[:one_npy.find('.')]] = 'high'
                    elif malignancy <= 2.5:
                        data_dic_low[one_npy[:one_npy.find('.')]] = 'low'
                    else:
                        data_dic_mid[one_npy[:one_npy.find('.')]] = 'mid'
                    # sphercity.append([float(onenodule[24]), float(onenodule[24])])
                    # margin.append([float(onenodule[25]), float(onenodule[25])])
                    # lobulation.append([float(onenodule[26]), float(onenodule[26])])
                    # spiculation.append([float(onenodule[27]), float(onenodule[27])])
                    

print('number of low malignant nodules: ', len(data_dic_low.keys()))
print('number of high malignant nodules: ', len(data_dic_high.keys()))
print('number of mid malignant nodules: ', len(data_dic_mid.keys()))

import random
low_data = []
high_data = []
mid_data = []

for one_key in data_dic_low.keys():
    temp = one_key + ',' + data_dic_low[one_key]
    low_data.append(temp)

for one_key in data_dic_high.keys():
    temp = one_key + ',' + data_dic_high[one_key]
    high_data.append(temp)

for one_key in data_dic_mid.keys():
    temp = one_key + ',' + data_dic_mid[one_key]
    mid_data.append(temp)

random.shuffle(low_data)
random.shuffle(high_data)
random.shuffle(mid_data)

csvTools.writeTXT('mid.csv',mid_data)

low_sign = len(low_data) // 5
high_sign = len(high_data) // 5
print(low_sign)
print(high_sign)

low_cross1 = low_data[:low_sign]
low_cross2 = low_data[low_sign:low_sign*2]
low_cross3 = low_data[low_sign*2:low_sign*3]
low_cross4 = low_data[low_sign*3:low_sign*4]
low_cross5 = low_data[low_sign*4:]

high_cross1 = high_data[:high_sign]
high_cross2 = high_data[high_sign:high_sign*2]
high_cross3 = high_data[high_sign*2:high_sign*3]
high_cross4 = high_data[high_sign*3:high_sign*4]
high_cross5 = high_data[high_sign*4:]

cross1 = low_cross1 + high_cross1
cross2 = low_cross2 + high_cross2
cross3 = low_cross3 + high_cross3
cross4 = low_cross4 + high_cross4
cross5 = low_cross5 + high_cross5

random.shuffle(cross1)
random.shuffle(cross2)
random.shuffle(cross3)
random.shuffle(cross4)
random.shuffle(cross5)

csvTools.writeTXT('cross1.csv',cross1)
csvTools.writeTXT('cross2.csv',cross2)
csvTools.writeTXT('cross3.csv',cross3)
csvTools.writeTXT('cross4.csv',cross4)
csvTools.writeTXT('cross5.csv',cross5)
