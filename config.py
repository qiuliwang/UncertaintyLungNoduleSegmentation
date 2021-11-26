import os

class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # about the model architecture
        self.arch = 'UIS+Channel'
        # training settings
        self.epochs = 300
        self.learning_rate = 0.0001
        self.gpu = 1
        self.evaluate = False # test or train
        self.resume = False
        self.num_classes = 3
        self.out_dim = 1
        self.lr_type = 'SGDR'
        self.milestones = [80, 160, 240]
        self.sgdr_t = 50
        self.weight_seg = 1
        self.weight_con = 0.1
        self.weight_im = 0.9
        self.weight_kd = 0.5

        # cross validation settings
        self.fold = 1
        self.fold_num = 5

        self.training_fold_index = []
        for i in range(self.fold_num + 1):
            if i != self.fold and i != 0:
                self.training_fold_index.append(i)

        self.test_fold_index = [self.fold]

        # paths 
        self.rowPath = '/home/ws/yanghan/Newcode/data/row_image/' # 对比实验所用数据集
        self.csvPath = './dataprocess/split_csv/' # fold_csv 
        self.llowerPath = './data/llower_image/'
        self.lowerPath = './data/lower_image/'
        self.midPath = './data/mid_image/'
        self.upperPath = './data/upper_image/'
        self.uupperPath = './data/uupper_image/'
        self.figurePath = './result/figure/'



        # if not os.path.exists(self.modelPath):
        #     os.makedirs(self.modelPath)