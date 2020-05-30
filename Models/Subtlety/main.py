from dataprepare import Data
from config import Config
import tensorflow as tf
import os
import numpy as np 
from tqdm import tqdm
from newmodel import RCNNMODEL
import csvTools
import random
import tensorflow.contrib.slim as slim
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
data = Data()

checkpoint_file = 'inception_v3_2016_08_28/inception_v3.ckpt'  
traintxt = csvTools.readCSV('./labels/cross2.csv') + csvTools.readCSV('./labels/cross3.csv') + csvTools.readCSV('./labels/cross4.csv') + csvTools.readCSV('./labels/cross5.csv')
testtxt = csvTools.readCSV('./labels/cross1.csv') 
validtxt = csvTools.readCSV('./labels/cross1.csv')

traindata = []
for one in traintxt:
    if len(one) != 0:
        
        traindata.append(one)

testdata=[]
for one in testtxt:
    if len(one) != 0:
        
        testdata.append(one)

validdata = []
for one in validtxt:
    if len(one) != 0:
        validdata.append(one)


'''
Prepare all data, rotating is here
'''

train_data, train_label, valid_data, valid_label, test_data, test_label = data.get_all_data(traindata, validdata, testdata)

print('number of train data: ', len(train_data))
print('number of valid data: ', len(valid_data))
print('number of test data: ', len(test_data))

print(train_data[0].shape)

config = Config()

load = False
load_cnn = False

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
# with tf.Session() as sess:

    model = RCNNMODEL(config)

    sess.run(tf.global_variables_initializer())   
    if load:
        model.load(sess, './models/16200.npy')
    if load_cnn:
        model.load_cnn(sess, './resnet50_no_fc.npy')


        # tf.global_variables_initializer().run()                                              
    #     # exclusions = ['Mixed_7c','Mixed_7b','AuxLogits','AuxLogits','Logits','Predictions','global_step']
    #     # variables_to_restore = slim.get_variables_to_restore(exclude=exclusions)
    #     # init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore, ignore_missing_vars=True)
    #     # init_fn(sess)        

    tf.get_default_graph().finalize()
    
    model.train(sess, train_data, train_label, valid_data, valid_label, data, False)


