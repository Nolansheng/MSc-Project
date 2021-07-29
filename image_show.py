import numpy as np
import matplotlib.pyplot as plt
import torchvision
import pandas as pd


'''
    to show the input image
'''
def map_normal2inital(img):
    for i in range(img.shape[0]):
        img[i] = img[i]*0.2 + 0.5 # unnormalize
    img = np.transpose((img * 255).astype(np.uint8), (0, 2, 3, 1))
    return img

def map_normal2inital2(img):
    for i in range(img.shape[0]):
        img[i] = img[i]*0.2 + 0.5 # unnormalize
    img = np.transpose((img * 255).numpy().astype(np.uint8), (0, 2, 3, 1))
    return img





'''This converts predicted map to RGB labels'''

class_path = 'CamVid/class_dict.csv'
classes = pd.read_csv(class_path, index_col=0)
label_colors = {cl: list(classes.loc[cl, :]) for cl in classes.index}
class_map = list(label_colors.values())

# from 3 channel to 2D image
def form_2D_label(mask):
    mask = mask.astype("uint8")
    label = np.zeros(mask.shape[:2], dtype=np.uint8)

    for i, rgb in enumerate(class_map):
        label[(mask == rgb).all(axis=2)] = i

    return label

# from 2D image to 32 channel
def adjust_channel(mask, channel_num=32):
    mask_size = mask.shape
    adjusted_mask = np.zeros((mask_size[0], mask_size[1], channel_num), dtype=int)

    for i in range(mask_size[0]):
        for j in range(mask_size[1]):
            class_no = mask[i][j]
            adjusted_mask[i][j][class_no] = 1
    return adjusted_mask

# from 32 channel to RGB image
def map32_rgb(label):
    y_pred = np.argmax(label, axis=1) # batch*H*W between 0-31
    y_pred_rgb = np.zeros((y_pred.shape[0], y_pred.shape[1], y_pred.shape[2], 3))
    for i in range(y_pred.shape[0]):
        image = np.zeros((y_pred.shape[1], y_pred.shape[2], 3))
        for j in range(y_pred.shape[1]):
            for k in range(y_pred.shape[2]):
                image[j, k, :] = class_map[y_pred[i, j, k]]
        y_pred_rgb[i] = image
    y_pred_rgb = np.array(y_pred_rgb, dtype='uint8')
    return y_pred_rgb

def show_image_label(image, label, batch_size):
  image = map_normal2inital2(image)
  label = map32_rgb(label)

  plt.figure(figsize=(16,9))
  for i in range(batch_size):
    plt.subplot(2, batch_size, i+1)
    plt.imshow(image[i])
    plt.subplot(2, batch_size, i+1+batch_size)
    plt.imshow(label[i])


def show_image_label_predict(images, labels, predicts, batch_size):
  image = map_normal2inital(images)
  label = map32_rgb(labels)
  predict = map32_rgb(predicts)

  plt.figure(figsize=(16,9))
  for i in range(batch_size):
    plt.subplot(3, batch_size, i+1)
    plt.imshow(image[i])
    plt.subplot(3, batch_size, i+1+batch_size)
    plt.imshow(label[i])
    plt.subplot(3, batch_size, i+1+2*batch_size)
    plt.imshow(predict[i])
