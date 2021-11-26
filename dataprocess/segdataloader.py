import os
import numpy as np
from glob import glob
from PIL import Image
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from Transforms import Scale
import cv2

# width, height = 96, 96


def att_compare(a, b=3):
    if a > b:
        return np.array([1])
    else:
        return np.array([0])


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_red, reds, blues, difs, masks, label=None, width=64, height=64):
        self.size = (width, height)
        self.data_red = data_red
        self.label = label
        self.reds = reds
        self.blues = blues
        self.difs = difs
        self.masks = masks
        self.img_resize = Compose([
            Scale(self.size, Image.BILINEAR),

        ])
        self.label_resize = Compose([
            Scale(self.size, Image.NEAREST),
        ])
        self.img_transform_gray = Compose([
            ToTensor(),
            Normalize(mean=[0.448749], std=[0.399953])  # 这个归一化方式可以提升近1个点的dice
        ])

        self.img_transform_rgb = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.input_red_paths = self.data_red

        self.red_path = self.reds
        self.blue_path = self.blues
        self.dif_path = self.difs
        self.mask_path = self.masks
        self.label_paths = self.label

        print('Training data:')
        print(len(self.data_red))
        print('Training Red Areas:')
        print(len(self.reds))
        print('Training Blue Areas:')
        print(len(self.blues))       
        print('Training Dif Areas:')
        print(len(self.difs))

        self.label_paths = self.label

    def __getitem__(self, index):
        image_red = Image.open(self.input_red_paths[index])
        red = Image.open(self.red_path[index])
        blue = Image.open(self.blue_path[index])
        dif = Image.open(self.dif_path[index])
        mask = Image.open(self.mask_path[index])

        image_red = self.img_resize(image_red)
        red = self.img_resize(red)
        blue = self.img_resize(blue)
        dif = self.img_resize(dif)
        mask = self.img_resize(mask)

        image_red = self.img_transform_gray(image_red)
        red = self.img_transform_gray(red)
        blue = self.img_transform_gray(blue)
        dif = self.img_transform_gray(dif)
        mask = self.img_transform_gray(mask)
        return image_red, red, blue, dif, mask

    def __len__(self):
        return len(self.red_path)


class Dataset_val(torch.utils.data.Dataset):
    def __init__(self, data_red, data_dif, reds, blues, difs, label=None, width=64, height=64):
        self.size = (width, height)
        self.data_red = data_red
        self.data_dif = data_dif
        self.label = label
        self.reds = reds
        self.blues = blues
        self.difs = difs
        self.img_resize = Compose([
            Scale(self.size, Image.BILINEAR),

        ])
        self.label_resize = Compose([
            Scale(self.size, Image.NEAREST),
        ])
        self.img_transform_gray = Compose([
            ToTensor(),
            Normalize(mean=[0.448749], std=[0.399953])  # 这个归一化方式可以提升近1个点的dice
        ])

        self.img_transform_rgb = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.input_red_paths = self.data_red
        self.input_dif_paths = self.data_dif

        self.red_path = self.reds
        self.blue_path = self.blues
        self.dif_path = self.difs
        self.label_paths = self.label
        print('Training data:')
        print(len(self.data_red))
        print('Training Red Areas:')
        print(len(self.reds))
        print('Training Blue Areas:')
        print(len(self.blues))       
        print('Training Dif Areas:')
        print(len(self.difs))

        self.label_paths = self.label

    def __getitem__(self, index):
        image_red = Image.open(self.input_red_paths[index])
        image_dif = Image.open(self.input_dif_paths[index])
        red = Image.open(self.red_path[index])
        blue = Image.open(self.blue_path[index])
        dif = Image.open(self.dif_path[index])

        image_red = self.img_resize(image_red)
        image_dif = self.img_resize(image_dif)
        red = self.img_resize(red)
        blue = self.img_resize(blue)
        dif = self.img_resize(dif)

        image_red = self.img_transform_gray(image_red)
        image_dif = self.img_transform_gray(image_dif)
        red = self.img_transform_gray(red)
        blue = self.img_transform_gray(blue)
        dif = self.img_transform_gray(dif)
        return image_red, image_dif, red, blue, dif

    def __len__(self):
        return len(self.red_path)


class RowDataset(torch.utils.data.Dataset):
    def __init__(self, data, mask, label=None, width=64, height=64):
        self.size = (width, height)
        self.data = data
        self.mask = mask
        self.label = label
        self.img_resize = Compose([
            Scale(self.size, Image.BILINEAR),

        ])
        self.label_resize = Compose([
            Scale(self.size, Image.NEAREST),
        ])
        self.img_transform_gray = Compose([
            ToTensor(),
            Normalize(mean=[0.448749], std=[0.399953])  # 这个归一化方式可以提升近1个点的dice
        ])

        self.img_transform_rgb = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.input_paths = self.data
        self.mask_paths = self.mask
        self.label_paths = self.label

    def __getitem__(self, index):
        image = Image.open(self.input_paths[index])
        mask = Image.open(self.mask_paths[index])
        # mask.save('image1.jpg')

        image = self.img_resize(image)
        mask = self.img_resize(mask)
        # mask.save('image2.jpg')

        image = self.img_transform_gray(image)
        mask = self.img_transform_gray(mask)

        # mask = mask.squeeze()
        # mask = np.transpose(mask.cpu().detach().numpy(), (0,1))
        # cv2.imwrite('image3.jpg', mask)
        return image, mask

    def __len__(self):
        return len(self.input_paths)


def loader(dataset, batch_size, num_workers=8, shuffle=False):
    input_images = dataset

    input_loader = torch.utils.data.DataLoader(dataset=input_images,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers)

    return input_loader
