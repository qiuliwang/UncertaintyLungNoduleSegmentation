import tensorflow as tf
import config
import pathlib
from config import image_height, image_width, channels
import csvTools
import os
import numpy as np 



def truncate_hu(image_array, max = 400, min = -900):
    image = image_array
    image[image > max] = max
    image[image < min] = min
    return image

def normalization(image_array):
    image_array = image_array + 900

    max = 1300.0
    min = 0
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    # avg = image_array.mean()
    # image_array = image_array-avg
    image_array = image_array * 255
    # image_array.dtype = 'int16'
    return image_array   # a bug here, a array must be returned,directly appling function did't work

def load_and_preprocess_image(img_path):
    '''
    Modified by Wang Qiuli
    2020/5.27
    '''

    # read pictures
    # decode pictures    
    # print(tf.shape(img_path))
    # print(img_path)
    img_tensor = tf.expand_dims(img_path, -1)
    # print(tf.shape(img_tensor))
    # print(img_tensor)
    # resize
    img_tensor = tf.image.resize(img_tensor, [image_height, image_width])
    # print(img_tensor)

    img_tensor = tf.cast(img_tensor, tf.float32)
    # print(img_tensor)
    # normalization
    img = img_tensor / 255.0
    return img

def get_images_and_labels(data_root_dir):
    '''
    Modified by WangQL
    2020/5/27
    '''
    csv = os.listdir(data_root_dir)
    data = []
    for one_csv in csv:
        data += csvTools.readCSV(os.path.join(data_root_dir, one_csv))

    all_image_path = []
    all_image_label = []

    data_path = 'D:/Data/nodules/ori_hu/'
    npy_files = os.listdir(data_path)

    for one_data in data:
        id = one_data[0]
        if 'low' in one_data[1]:
            label = [0]
        else:
            label = [1]
        # print(id)
        npy = np.load(os.path.join(data_path, one_data[0] + '.npy'))
        npy = truncate_hu(npy)
        nor_npy = normalization(npy)
        # print(nor_npy)
        all_image_path.append(nor_npy)
        all_image_label.append(label)
        if 'high' in one_data[1]:
            all_image_path.append(nor_npy)
            all_image_label.append(label)

    # # get all images' paths (format: string)
    # data_root = pathlib.Path(data_root_dir)
    # all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # # get labels' names
    # label_names = sorted(item.name for item in data_root.glob('*/'))
    # # dict: {label : index}
    # label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # # get all images' labels
    # all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]

    # return all_image_path, all_image_label
    # # get all images' paths (format: string)
    # data_root = pathlib.Path(data_root_dir)
    # all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # # get labels' names
    # label_names = sorted(item.name for item in data_root.glob('*/'))
    # # dict: {label : index}
    # label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # # get all images' labels
    # all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]

    return all_image_path, all_image_label



def get_dataset(dataset_root_dir):
    all_image_path, all_image_label = get_images_and_labels(data_root_dir=dataset_root_dir)
    # print("image_path: {}".format(all_image_path[:]))
    # print("image_label: {}".format(all_image_label[:]))
    # load the dataset and preprocess images
    image_dataset = tf.data.Dataset.from_tensor_slices(all_image_path).map(load_and_preprocess_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    image_count = len(all_image_path)

    return dataset, image_count


def generate_datasets():
    train_dataset, train_count = get_dataset(dataset_root_dir=config.train_dir)
    valid_dataset, valid_count = get_dataset(dataset_root_dir=config.valid_dir)
    test_dataset, test_count = get_dataset(dataset_root_dir=config.test_dir)


    # read the original_dataset in the form of batch
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=config.BATCH_SIZE)

    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count
