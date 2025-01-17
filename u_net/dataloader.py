import os
from os.path import isdir, exists, abspath, join
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

class DataLoader():
    def __init__(self, root_dir='data/cells/', batch_size=1, test_percent=.3):
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
        indexes = [i for i in range(current, endId)]
        # random.shuffle(indexes)        
        for current in indexes:
            # todo: load images and labels
            # hint: scale images between 0 and 1
            # hint: if training takes too long or memory overflow, reduce image size!
            resized_size = 572
            # load images
            data_path = self.data_files[current]
            # print("data_path = ", data_path)
            data_image = Image.open(data_path)
            data_image = data_image.resize((resized_size, resized_size))

            # load labels
            label_path = self.label_files[current]
            label_image = Image.open(label_path)
            label_image = label_image.resize((resized_size, resized_size))

            # apply data augmentation and normalization
            data_image, label_image = self.__applyDataAugmentation(data_image, label_image)

            yield (data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))

    # data augumentation function
    def __applyDataAugmentation(self, data_image, label_image):
        # flip {0: none, 1: horizontal, 2: vertical}
        flipOption = random.randint(0,2)
        # zoom {0: none, 1: 1/0.95, 2: 1/0.9}
        zoomOption = random.randint(0,2)
        # rotate {0: 0, 1: 90, 2: 180}            
        rotateOption = random.randint(0,2)
        # gamma {0: 0, 1: 0.8}            
        gammaOption = random.randint(0,1)
        # elastic {0: none, 1: distort} 
        elasticOption = random.randint(0,1)

        data_image = self.__flip(data_image, flipOption)
        label_image = self.__flip(label_image, flipOption)
        
        data_image = self.__zoom(data_image, zoomOption)
        label_image = self.__zoom(label_image, zoomOption)
        
        data_image = self.__rotate(data_image, rotateOption)
        label_image = self.__rotate(label_image, rotateOption)

        data_image = self.__gamma(data_image, gammaOption)
        
        # normalization
        data_image = np.asarray(data_image, dtype=np.float32) / 255.
        # min_, max_ = float(np.min(data_image)), float(np.max(data_image))
        # data_image = (data_image - min_) / (max_ - min_)  
        label_image = np.asarray(label_image, dtype=np.float32)
        
        data_image = self.__elastic_deform(data_image, elasticOption)
        label_image = self.__elastic_deform(label_image, elasticOption)
        
        return data_image, label_image
            
    def __flip(self, image, flipOption):
        if flipOption == 1:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif flipOption == 2:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def __zoom(self, image, zoomOption):
        resized_size, _ = image.size
        crop_ratio = 1
        if zoomOption == 1:
            crop_ratio = 0.95
        elif zoomOption == 2:
            crop_ratio = 0.9
        crop_start = int(resized_size*(1-crop_ratio)/2)
        crop_size = int(resized_size*crop_ratio)
        crop_pos= (crop_start,crop_start,crop_start+crop_size,crop_start+crop_size)
        image = image.crop(crop_pos).resize((resized_size, resized_size))
        return image

    def __rotate(self, image, rotateOption):
        if rotateOption == 1:
            image = image.transpose(Image.ROTATE_90)
        elif rotateOption == 2:
            image = image.transpose(Image.ROTATE_180)
        return image

    def __gamma(self, image, gammaOption):
        gamma = 1
        if gammaOption == 1:
            gamma = 0.8
        image = transforms.functional.adjust_gamma(image, gamma, gain=1)
        return image

    def __elastic_deform(self, image, elasticOption):
        if elasticOption == 1:
            alpha=34
            sigma = random.randint(6, 12)
            random_state = np.random.RandomState(None)
            shape = image.shape
            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            image = map_coordinates(image, indices, order=1, mode='reflect').reshape(image.shape)
        return image


# Test dataloader
# loader = DataLoader('data/cells/')
# for i, (img, label) in enumerate(loader):
#    figs, axes = plt.subplots(1, 2)
#    print(img.shape)
#    axes[0].imshow(img)
#    axes[1].imshow(label)
#    plt.show()
#    if(i==3): 
#        break

