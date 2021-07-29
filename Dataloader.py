import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import os
from glob import glob
import pandas as pd
import numpy as np
from numpy import moveaxis
from torchvision import transforms, utils, datasets
from PIL import Image




# classes in each picture
class_path = 'CamVid/class_dict.csv'
classes = pd.read_csv(class_path, index_col=0)
label_colors = {cl: list(classes.loc[cl, :]) for cl in classes.index}
class_map = list(label_colors.values())


'''This method will convert mask labels(to be trained) from RGB to a 2D image which holds class labels of the pixels.'''


def form_2D_label(mask, class_maps=class_map):
    mask = mask.astype("uint8")
    label = np.zeros(mask.shape[:2], dtype=np.uint8)

    for i, rgb in enumerate(class_maps):
        label[(mask == rgb).all(axis=2)] = i

    return label


'''from 2D image to 32 channel image base on the class map'''
def adjust_channel(mask, channel_num=32):
    mask_size = mask.shape
    adjusted_mask = np.zeros((mask_size[0], mask_size[1], channel_num), dtype=np.uint8)
    for i in range(mask_size[0]):
        for j in range(mask_size[1]):
            class_no = mask[i][j]
            adjusted_mask[i][j][class_no] = 1
    return adjusted_mask


# y_pred = np.argmax(y_pred, axis=2)  # axis=3
'''This converts predicted map to RGB labels'''
# def map_this(y_pred,class_map):
#     y_pred_rgb = np.zeros((y_pred.shape[0],y_pred.shape[1],y_pred.shape[2],3))
#     for i in range(y_pred.shape[0]):
#         image = np.zeros((y_pred.shape[1],y_pred.shape[2],3))
#         for j in range(y_pred.shape[1]):
#             for k in range(y_pred.shape[2]):
#                 image[j,k,:] = class_map[y_pred[i][j][k]]
#         y_pred_rgb[i] = image
#     return y_pred_rgb






from torchvision import transforms as T
transform_t = T.Compose([
 T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
 T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]) # 标准化至[-1, 1]，规定均值和标准差
])


class LoadDataset(Dataset): #继承Dataset
    def __init__(self, root='CamVid', train_or_val='train', transform=transform_t): #初始化一些属性
        if train_or_val == 'train':
            self.img_files = glob(os.path.join(root, 'train', '*.png'))
            self.img_files.sort()  # sort the files with names
            self.mask_files = glob(os.path.join(root, 'train_labels', '*.png'))
            self.mask_files.sort()
        elif train_or_val == 'val':
            self.img_files = glob(os.path.join(root, 'val', '*.png'))
            self.img_files.sort()  # sort the files with names
            self.mask_files = glob(os.path.join(root, 'val_labels', '*.png'))
            self.mask_files.sort()
        else:
            self.img_files = glob(os.path.join(root, 'test', '*.png'))
            self.img_files.sort()  # sort the files with names
            self.mask_files = glob(os.path.join(root, 'test_labels', '*.png'))
            self.mask_files.sort()

        self.transform = transform #对图形进行处理，如标准化、截取、转换等

    def __len__(self):#返回整个数据集的大小
        return len(self.img_files)

    def __getitem__(self, index):#根据索引index返回图像及标签
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        # img = Image.open(img_path).convert('RGB')# 读取图像
        img = cv2.imread(img_path,  cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor((img).astype(np.uint8), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        if self.transform is not None:
            img = self.transform(img)

        label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        label = cv2.cvtColor((label).astype(np.uint8), cv2.COLOR_BGR2RGB)
        label = cv2.resize(label, (256, 256))
        label_2D = form_2D_label(label)
        new_label = adjust_channel(label_2D)
        new_label = moveaxis(new_label, 2, 0)
        return img, torch.from_numpy(new_label).float()







