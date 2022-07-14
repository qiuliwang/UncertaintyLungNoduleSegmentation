import os

from numpy.core.numeric import False_

class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # about the model architecture
        self.arch = 'U_Net_ori'
        # self.arch = 'U_Net_grey'
        # training settings
        self.epochs = 150
        self.learning_rate = 0.0001
        self.gpu = 7
        self.evaluate = True # test or train
        self.resume = False
        self.num_classes = 3
        self.in_dim = 3
        self.out_dim = 1
        self.lr_type = 'SGDR'
        self.milestones = [80, 160, 240]
        self.sgdr_t = 50
        self.weight_seg1 = 1
        self.weight_seg2 = 0.5
        self.weight_con = 0.1
        self.weight_im = 0.9
        self.weight_kd = 0.5
        self.weight_edge = 0.5
        self.batch_size = 32

        # cross validation settings
        self.fold = 1
        self.fold_num = 5

        self.training_fold_index = []
        for i in range(self.fold_num + 1):
            if i != self.fold and i != 0:
                self.training_fold_index.append(i)

        self.test_fold_index = [self.fold]

        # paths 
        self.mask_path = '/home1/qiuliwang/Code/wsi_extractor_python/Glioma_Extracted_Patch_512/'
        self.image_path = '/home1/qiuliwang/Code/wsi_extractor_python/Glioma_Extracted_Patch_512/'
        self.csvPath = '/home1/qiuliwang/Code/wsi_extractor_python/PrepareDatasetCsv/' # fold_csv 

        self.figurePath = './result/figure/'



        # if not os.path.exists(self.modelPath):
        #     os.makedirs(self.modelPath)