import os
from os.path import isdir, exists, abspath, join

import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.1):
        self.mode = 'train'
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        while current < endId:
            current += 1
            # todo: load images and labels
            # hint: scale images between 0 and 1
            # hint: if training takes too long or memory overflow, reduce image size!
            resized_size = 300
            # load images
            data_path = self.data_files[current]
            # print("data_path = ", data_path)
            data_image = Image.open(data_path)
            data_image = data_image.resize((resized_size, resized_size))

            # load labels
            label_path = self.label_files[current]
            label_image = Image.open(label_path)
            label_image = label_image.resize((resized_size, resized_size))

            # data augumentation: flip, zoom, rotate, gamma correction
            will_flip = random.uniform(0, 1)
            will_zoom = random.uniform(0, 1)
            will_rotate = random.uniform(0, 1)
            will_gamma = random.uniform(0, 1)

            crop_ratio = 0.8
            crop_start = resized_size*(1-crop_ratio)/2
            crop_size = resized_size*crop_ratio
            gamma = 2.2

            if will_flip > 0.5:
                data_image = data_image.transpose(Image.FLIP_LEFT_RIGHT)
                label_image = label_image.transpose(Image.FLIP_LEFT_RIGHT)              
            if will_zoom > 0.5:
                crop_pos= (crop_start,crop_start,crop_start+crop_size,crop_start+crop_size)
                data_image = data_image.crop(crop_pos).resize((resized_size, resized_size))
                label_image = label_image.crop(crop_pos).resize((resized_size, resized_size))
            if will_rotate > 0.5:
                data_image = data_image.transpose(Image.ROTATE_90)
                label_image = label_image.transpose(Image.ROTATE_90)

            data_image = np.asarray(data_image, dtype=np.float32)
            min_, max_ = float(np.min(data_image)), float(np.max(data_image))
            data_image = (data_image - min_) / (max_ - min_)  # normalization
            label_image = np.asarray(label_image, dtype=np.float32)

            if will_gamma > 0.5:
                data_image = data_image ** (1 / gamma) * 255
                label_image = label_image ** (1 / gamma) * 255

            yield (data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))


# Test dataloader
# loader = DataLoader('data/cells/')
# for i, (img, label) in enumerate(loader):
#     figs, axes = plt.subplots(1, 2)
#     axes[0].imshow(img)
#     axes[1].imshow(label)
#     plt.show()


