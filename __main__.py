from Rename import rename
import image_show
from Dataloader import LoadDataset
from torchvision.utils import save_image
import numpy as np
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import torch
from ResNet import ResNet
from ResNet import Bottleneck
import torch.nn as nn
from PSPNet import PSPNet
import os
import torch.nn.functional as F
import torchvision
from Upsampling import Upsample
from PSPNet import PSPNet

if __name__ == '__main__':

    # train_path = 'CamVid/train'
    # train_label_path = 'CamVid/train_labels'
    #
    # test_path = 'CamVid/test'
    # test_label_path = 'CamVid/test_labels'
    #
    # val_path = 'CamVid/val'
    # val_label_path = 'CamVid/val_labels'
    #
    # rename_train = rename(train_path, train_label_path)
    # rename_test = rename(test_path, test_label_path)
    # rename_val = rename(val_path, val_label_path)
    #
    # rename_train.train_rename()
    # rename_test.test_rename()
    # rename_val.val_rename()

    model = PSPNet()
    # setting of learning rate, optimizer and loss function
    input = torch.rand(4, 3, 720, 960)
    model.eval()
    print(model)
    output = model(input)
    print('PSPNet', output.size())
    # learning_rate = 0.001
    # Loss = nn.NLLLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #
    # train_set = LoadDataset(train_or_val='train')
    # num_workers = 4
    # batch_size = 10
    # train_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    #
    # val_set = LoadDataset(train_or_val='val')
    # val_data_loader = DataLoader(dataset=val_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    #
    # num_epochs = 1
    # for epoch in range(num_epochs):
    #     print('Epoch:', epoch)
    #     model.train()
    #     for iteration, sample in enumerate(train_data_loader):
    #         img, mask = sample
    #
    #         y_predict = model.forward(img)
    #         print(y_predict.shape)
    #         target = torch.argmax(mask, dim=1)
    #         loss = Loss(y_predict, target)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         break


    # for sample in train_data_loader:
    #     img, mask = sample
    #     img1 = img[0, ...].squeeze()
    #     print(img.shape)
    #     print(mask.shape)
        # msk1 = mask[0, ...].squeeze()
        # msk1 =msk1.numpy()
        # print(msk1.shape)
        #
        # break

    # to show the original images and masks
    # dataiter = iter(train_data_loader)
    # images, labels = dataiter.next()
    # y_pred_rgb = image_show.map32_rgb(labels)
    # initial_image = image_show.map_normal2inital(images)
    # image_show.show_image_label(initial_image, y_pred_rgb, batch_size)





