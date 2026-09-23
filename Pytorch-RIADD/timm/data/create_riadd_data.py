from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import os
import re
import torch
import tarfile
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from timm.data.riadd_augment import crop_maskImg

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']

class RiaddDataSet(data.Dataset):  # for training/testing
    def __init__(self, image_ids,baseImgPath = '',selftrans = False, load_bytes=False,transform=None,onlydisease = False):
        self.image_ids = image_ids
        self.baseImgPath = baseImgPath
        self.transform = transform
        self.load_bytes = load_bytes
        self.selftrans = selftrans
        self.onlydisease = onlydisease
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        imgId = self.image_ids.iloc[index, 0]
        imgId = str(imgId) + '.png'
        if self.onlydisease == True:
            label_2 = self.image_ids.iloc[index, 2:].values.astype(np.int64)
            label_2 = sum(label_2)
            label = self.image_ids.iloc[index, 1:2].values.astype(np.int64)
            if label_2 > 0:
                label = np.append(label, 1)
            else:
                label = np.append(label,0)

        else:
            label = self.image_ids.iloc[index, 1:].values.astype(np.int64)
        imgpath = os.path.join(self.baseImgPath,imgId)
        if self.selftrans == True:
            img = open(path, 'rb').read() if self.load_bytes else Image.open(imgpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = cv2.imread(imgpath)
            img = crop_maskImg(img)
            img = img[:, :, ::-1]
            img = self.transform(image = img)['image']
        return img,label


class RiaddDataSet11Classes(data.Dataset):  # for training/testing
    def __init__(self, image_ids,baseImgPath = '',selftrans = False, load_bytes=False,transform=None,onlydisease = False):
        self.image_ids = image_ids
        self.baseImgPath = baseImgPath
        self.transform = transform
        self.load_bytes = load_bytes
        self.selftrans = selftrans
        self.onlydisease = onlydisease
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        imgId = self.image_ids.iloc[index, 0]
        imgId = str(imgId) + '.png'
        if self.onlydisease == True:
            label = self.image_ids.iloc[index, 1:2].values.astype(np.int64)
        else:
            label = self.image_ids.iloc[index, [9,11,15,16,19,20,21,22,25,27,28]].values.astype(np.int64)
        imgpath = os.path.join(self.baseImgPath,imgId)
        if self.selftrans == True:
            img = open(path, 'rb').read() if self.load_bytes else Image.open(imgpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = cv2.imread(imgpath)
            img = crop_maskImg(img)
            img = img[:, :, ::-1]
            img = self.transform(image = img)['image']
        return img,label

class RiaddDataSet9Classes(data.Dataset):  # for training/testing
    def __init__(self, image_ids,baseImgPath = '',selftrans = False, load_bytes=False,transform=None,onlydisease = False):
        self.image_ids = image_ids
        self.baseImgPath = baseImgPath
        self.transform = transform
        self.load_bytes = load_bytes
        self.selftrans = selftrans
        self.onlydisease = onlydisease
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        imgId = self.image_ids.iloc[index, 0]
        imgId = str(imgId) + '.png'
        if self.onlydisease == True:
            label = self.image_ids.iloc[index, 1:2].values.astype(np.int64)
        else:
            label = self.image_ids.iloc[index, [2,3,4,5,6,7,8,13,17]].values.astype(np.int64)
        imgpath = os.path.join(self.baseImgPath,imgId)
        if self.selftrans == True:
            img = open(path, 'rb').read() if self.load_bytes else Image.open(imgpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = cv2.imread(imgpath)
            img = crop_maskImg(img)
            img = img[:, :, ::-1]
            img = self.transform(image = img)['image']
        return img,label

class RiaddDataSet8Classes(data.Dataset):  # for training/testing
    def __init__(self, image_ids,baseImgPath = '',selftrans = False, load_bytes=False,transform=None,onlydisease = False):
        self.image_ids = image_ids
        self.baseImgPath = baseImgPath
        self.transform = transform
        self.load_bytes = load_bytes
        self.selftrans = selftrans
        self.onlydisease = onlydisease
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        imgId = self.image_ids.iloc[index, 0]
        imgId = str(imgId) + '.png'
        if self.onlydisease == True:
            label = self.image_ids.iloc[index, 1:2].values.astype(np.int64)
        else:
            label = self.image_ids.iloc[index, [10,12,14,18,23,24,26,29]].values.astype(np.int64)
        imgpath = os.path.join(self.baseImgPath,imgId)
        if self.selftrans == True:
            img = open(path, 'rb').read() if self.load_bytes else Image.open(imgpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = cv2.imread(imgpath)
            img = crop_maskImg(img)
            img = img[:, :, ::-1]
            img = self.transform(image = img)['image']
        return img,label

class PrismDataSet(data.Dataset):  # for training/testing
    """RiaddDataSet for datasets whose CSV id column already carries the file
    extension (PRISM, MURED), instead of RFMiD's bare integer ids."""
    def __init__(self, image_ids, baseImgPath = '', selftrans = False, load_bytes=False, transform=None):
        self.image_ids = image_ids
        self.baseImgPath = baseImgPath
        self.transform = transform
        self.load_bytes = load_bytes
        self.selftrans = selftrans

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        imgId = str(self.image_ids.iloc[index, 0])
        label = self.image_ids.iloc[index, 1:].values.astype(np.int64)
        imgpath = os.path.join(self.baseImgPath, imgId)
        if self.selftrans == True:
            img = open(imgpath, 'rb').read() if self.load_bytes else Image.open(imgpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = cv2.imread(imgpath)
            img = crop_maskImg(img)
            img = img[:, :, ::-1]
            img = self.transform(image = img)['image']
        return img, label


class PrismDataSetGroup(data.Dataset):
    """PrismDataSet restricted to an explicit column-index group -- the MURED
    analogue of RiaddDataSet9/8/11Classes, whose iloc lists are hardcoded to
    RFMiD's 29-column layout and do not apply here. `cols` are iloc indices
    into image_ids (1-based: index 0 is the id column), matching how the
    RFMiD Classes variants index their own frequency tiers."""
    def __init__(self, image_ids, cols, baseImgPath='', transform=None):
        self.image_ids = image_ids
        self.cols = list(cols)
        self.baseImgPath = baseImgPath
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        imgId = str(self.image_ids.iloc[index, 0])
        label = self.image_ids.iloc[index, self.cols].values.astype(np.int64)
        imgpath = os.path.join(self.baseImgPath, imgId)
        img = cv2.imread(imgpath)
        img = crop_maskImg(img)
        img = img[:, :, ::-1]
        img = self.transform(image=img)['image']
        return img, label


class PrismDataSetNormal(data.Dataset):
    """MURED analogue of RiaddDataSet(onlydisease=True) -- RFMiD's 'dr' head
    predicts [Disease_Risk, any-of-the-other-columns], a 2-output redundant
    pair. MURED has no Disease_Risk column; NORMAL (iloc 2, positive =
    healthy) plays the equivalent role, so this predicts
    [NORMAL, any-of-the-other-19-columns] the same structural way."""
    NORMAL_COL = 2

    def __init__(self, image_ids, baseImgPath='', transform=None):
        self.image_ids = image_ids
        self.baseImgPath = baseImgPath
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        imgId = str(self.image_ids.iloc[index, 0])
        other_cols = [c for c in range(1, self.image_ids.shape[1]) if c != self.NORMAL_COL]
        any_other = int(self.image_ids.iloc[index, other_cols].values.astype(np.int64).sum() > 0)
        normal = int(self.image_ids.iloc[index, self.NORMAL_COL])
        label = np.array([normal, any_other], dtype=np.int64)
        imgpath = os.path.join(self.baseImgPath, imgId)
        img = cv2.imread(imgpath)
        img = crop_maskImg(img)
        img = img[:, :, ::-1]
        img = self.transform(image=img)['image']
        return img, label
