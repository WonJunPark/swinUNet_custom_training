# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:55
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : dataset.py
"""

"""
import os
import os.path as osp
import logging

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms



class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir   = imgs_dir
        self.masks_dir  = masks_dir
        self.scale      = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        # self.img_names = sorted(os.listdir(imgs_dir))
        # logging.info(f'Creating dataset with {len(self.img_names)} examples')

    def __len__(self):
        # return len(self.img_names)
        return len(self.imgs_dir)

    @classmethod
    def preprocess(cls, pil_img, scale):

        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))
        pil_img = cv2.resize(pil_img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        img_nd = np.array(pil_img)


        if len(img_nd.shape) == 2:
            # mask target image
            img_nd = np.expand_dims(img_nd, axis=2)
        else:
            # grayscale input image
            # scale between 0 and 1
            img_nd = img_nd / 255
        # HWC to CHW

        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans.astype(float)

    def __getitem__(self, i):
        # img_name = self.img_names[i]
        # img_path = osp.join(self.imgs_dir, img_name)
        # mask_path = osp.join(self.masks_dir, img_name)

        img_path = self.imgs_dir[i]
        mask_path = self.masks_dir[i]


        # img = Image.open(img_path)
        mask = Image.open(mask_path)
        # mask = cv2.imread(mask_path)
        img = cv2.imread(img_path)

        img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)

        # plt.imshow(mask)
        # plt.show()

        # mask = cv2.imread(mask_path)

        # assert img.size == mask.size, \
        #     f'Image and mask {img_name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
