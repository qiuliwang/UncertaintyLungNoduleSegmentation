import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import copy
import json
from tqdm import tqdm
import scipy.misc

from utils.nn import NN
import cv2
from utils.misc import ImageLoader, CaptionData, TopN
from sklearn.metrics import roc_auc_score,recall_score

def GetSenAndSpi(labels, predictions):
    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0

    for label, pred in zip(labels, predictions):
        if label[0] == 1 and pred == True:
            TP += 1
        elif label[0] == 1 and pred == False:
            FP += 1
        elif label[0] == 0 and pred == True:
            TN += 1
        elif label[0] == 0 and pred == False:
            FN += 1
    
    Sensitivity = (TP) / (TP + FP)
    Specificity = (TN) / (TN + FN)
    return Sensitivity, Specificity


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.train_cnn = self.is_train and config.train_cnn
        self.image_shape = [128, 128, 3]
        self.nn = NN(config)
        self.global_step = tf.Variable(0,
                                       name = 'global_step',
                                       trainable = False)
        self.build()
        self.record = open('lossrecord_oneloss.txt', 'w')
        self.predrecord = open('predrecord__oneloss'+ self.config.cnn + '_'+ str(self.config.num_lstm_units) + '_'+'_.csv', 'w')


    def build(self):
        raise NotImplementedError()

    def train(self, sess, train_data, train_label, test_data, test_label, dataobj, transign):
        """ Train the model. """
        print("Training the model...")
        config = self.config

        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)

        for _ in tqdm(list(range(config.num_epochs)), desc='epoch'):
            times = len(train_data) // config.batch_size
            if (len(train_data) % config.batch_size) != 0:
                times = times + 1
            
            for t in range(times):
                batch_files = np.array(train_data[t * config.batch_size:(t + 1) * config.batch_size])
                batch_label = np.array(train_label[t * config.batch_size:(t + 1) * config.batch_size])
                
                feed_dict = {self.images: batch_files, self.real_label: batch_label}            
                
                _, summary, global_step = sess.run([self.opt_op,
                                                    self.summary,
                                                    self.global_step],
                                                    feed_dict=feed_dict)

            feed_dict = {self.images: np.array(train_data[:256]), self.real_label: np.array(train_label[:256])}
            loss, correctpred = sess.run([self.loss, self.correct_pred], feed_dict=feed_dict)
            numberOfTrue = 0.0
            for one in correctpred:
                if one:
                    numberOfTrue += 1
            print('train loss: ', loss)
            print('train acc: ', numberOfTrue / len(correctpred))

            feed_dict = {self.images: np.array(test_data), self.real_label: np.array(test_label)}
            loss, correctpred = sess.run([self.loss, self.correct_pred], feed_dict=feed_dict)
            numberOfTrue = 0.0
            for one in correctpred:
                if one:
                    numberOfTrue += 1
            Sensitivity, Specificity = GetSenAndSpi(test_label, correctpred)

            print('valid loss: ', loss)
            print('valid acc: ', numberOfTrue / len(correctpred))
            print('valid Sen: ', Sensitivity)
            print('valid Spic', Specificity)

            train_writer.add_summary(summary, global_step)

        self.save()
        train_writer.close()
        print("Training complete.")

    def test_get_mid_images(self, sess, testdata, dataobj, sign):
        """ Test the model using any given images. """
        """ Gives out the middle outputs. """

        print("Testing the model ..." + sign)
        config = self.config
        label = []
        for onetestpatient in tqdm(testdata):
            if 'nor' in onetestpatient[1]:
                onelabel = [[0, 1]]
            else:
                onelabel = [[1, 0]]
            label.append(np.argmin(onelabel))
            slices = dataobj.getOnePatient(onetestpatient)            
            feed_dict = {self.images: slices, self.real_label: onelabel}
            middleconvs4 = sess.run([self.middleconvs], feed_dict=feed_dict)
            # print(middleconvs4[0].shape)
            # print(len(middleconvs4))
            patientid = onetestpatient[0]
            if 'nor' not in onetestpatient[1]:
                #(32, 128, 128, 256)
                middle = middleconvs4[0]
                for i in range(32):
                    temp_i = middle[i]
                    temp_image = temp_i[:,:,0]
                    if not os.path.exists('./midfeature/'+patientid):
                        os.mkdir('./midfeature/'+patientid)
                    for j in range(1,256):
                        oneconv = temp_i[:,:,j]
                        temp_image += oneconv
                    temp_image = cv2.resize(temp_image,(512,512))
                    scipy.misc.imsave('./midfeature/' + patientid +'/'+ str(i) + '.jpg', temp_image)


        print("Test completed.")

    def getLabel2(self, onepatient, alllabel):
        for onelabel in alllabel:
            if onepatient == onelabel[0]:
                if int(onelabel[1]) == 1:
                    return([0, 1])
                elif int(onelabel[1]) == 0:
                    return([1, 0])

                break
            
    def getLabel(self, onepatient, alllabel):
        label = []
        for onelabel in alllabel:
            if onepatient == onelabel[0]:
                if int(onelabel[1]) == 1:
                    label.append([0, 1])
                elif int(onelabel[1]) == 0:
                    label.append([1, 0])

                break
            
        return label

    def test(self, sess, testdata, dataobj, sign):
        """ Test the model using any given images. """
        print("Testing the model ..." + sign)
        config = self.config
        predrecord = self.predrecord
        predrecord.write('id,ret\n')
        correctpercent = 0
        TP=0
        TN=0
        FP=0
        FN=0
        label=[]
        pre=[]
        for onetestpatient in tqdm(testdata):
            if 'nor' in onetestpatient[1]:
                onelabel = [[0, 1]]
            else:
                onelabel = [[1, 0]]
            label.append(np.argmin(onelabel))

            slices = dataobj.getOnePatient(onetestpatient)            
            feed_dict = {self.images: slices, self.real_label: onelabel}
            prediction, correct_pred = sess.run([self.prediction, self.correct_pred], feed_dict=feed_dict)
            pre.append(1-prediction)
#            print(correct_pred)
            if correct_pred == True: 
                correctpercent += 1
                if 'nor' in onetestpatient[1]:
                    TN+=1
                else:
                    TP+=1
            else:
                if 'nor'  in onetestpatient[1]:
                    FP+=1
                else:
                    FN+=1
                predrecord.write(onetestpatient[0]+','+onetestpatient[1]+'\n')
#                print(onetestpatient)
        print(correctpercent)
        print(len(testdata))
        print('Sensitivity',float(TP)/float(TP+FN))
        print('Specificity',float(TN)/float(FP+TN))
        print('recall_score',recall_score(label,pre))
        print('AUC',roc_auc_score(label,pre))
        
        print('accuracy: ', float(correctpercent) / len(testdata)) 

        print("Test completed.")

    def save(self):
        """ Save the model. """
        config = self.config
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.save_dir, str(self.global_step.eval()))

        print((" Saving the model to %s..." % (save_path +".npy")))
        np.save(save_path, data)
        info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)
        info_file.close()
        print("Model saved.")

    def load(self, sess, model_file=None):
        """ Load the model. """
        config = self.config
        if model_file is not None:
            save_path = model_file
        else:
            info_path = os.path.join(config.save_dir, "config.pickle")
            info_file = open(info_path, "rb")
            config = pickle.load(info_file)
            global_step = config.global_step
            info_file.close()
            save_path = os.path.join(config.save_dir,
                                     str(global_step)+".npy")

        print("Loading the model from %s..." %save_path)
        data_dict = np.load(save_path).item()
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("%d tensors loaded." %count)
    
    # def load_tf(self, sess, )

    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading the CNN from %s..." %data_path)
        data_dict = np.load(data_path).item()
        count = 0
        for op_name in tqdm(data_dict):
            with tf.variable_scope(op_name, reuse = True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                    except ValueError:
                        pass
        print("%d tensors loaded." %count)
