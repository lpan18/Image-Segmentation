import sys
import os
from os.path import join
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from torchvision import transforms

import matplotlib.pyplot as plt

from model import UNet
from dataloader import DataLoader

from torch.autograd import Variable
import torch.nn.functional as F

WILL_TRAIN = True
WILL_TEST = False

def train_net(net,
              epochs=5,
              data_dir='data/cells/',
              n_classes=2,
              lr=0.1,
              val_percent=0.1,
              save_cp=True,
              gpu=False):
    loader = DataLoader(data_dir)

    N_train = loader.n_train()
 
    optimizer = optim.SGD(net.parameters(),
                            lr=lr,
                            momentum=0.99,
                            weight_decay=0.0005)

    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()
        loader.setMode('train')

        epoch_loss = 0

        for i, (img, label) in enumerate(loader):
            shape = img.shape
            label = label - 1
            # todo: create image tensor: (N,C,H,W) - (batch size=1,channels=1,height,width)
            img_torch = torch.from_numpy(img.reshape(1,1,shape[0],shape[1])).float()

            # todo: load image tensor to gpu
            if gpu:
                img_torch = Variable(img_torch.cuda())
            optimizer.zero_grad()
            # todo: get prediction and getLoss()
            pred_label = net(img_torch)
            # # crop target_label
            # crop_start = np.int64((shape[0] - pred_label.shape[2])/2)
            # crop_end = crop_start + pred_label.shape[2]
            # label = label[crop_start:crop_end, crop_start:crop_end]
            target_label = torch.from_numpy(label).float()

            if gpu:
                target_label = Variable(target_label.cuda())
            
            loss = getLoss(pred_label, target_label)
            epoch_loss += loss.item()
 
            print('Training sample %d / %d - Loss: %.6f' % (i+1, N_train, loss.item()))

            # optimize weights
            loss.backward()
            optimizer.step()

        if((epoch + 1) % 1 == 0) :   
            torch.save(net.state_dict(), join(data_dir, 'checkpoints') + '/CP%d.pth' % (epoch + 1))
            print('Checkpoint %d saved !' % (epoch + 1))
            print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / i))

# displays test images with original and predicted masks 
def test_net(testNet, 
            gpu=True,
            data_dir='data/cells/'):
    loader = DataLoader(data_dir)
    loader.setMode('test')
    testNet.eval()

    with torch.no_grad():
        for _, (img, label) in enumerate(loader):
            shape = img.shape
            img_torch = torch.from_numpy(img.reshape(1,1,shape[0],shape[1])).float()
            if gpu:
                img_torch = img_torch.cuda()
            pred = testNet(img_torch)
            pred_sm = softmax(pred)
            _,pred_label = torch.max(pred_sm,1)
            result = pred_label.cpu().detach().numpy().squeeze()*255
            # == test ==
            # print(result)
            # for i in range(pred_label.shape[0]):
            #     for j in range(pred_label.shape[0]):
            #         if pred_label[i][j] != 1:
            #             print("i=", i, " j=", j, "pred_label[i][j]", pred_label[i][j])
                
            plt.subplot(1, 3, 1)
            plt.imshow(img*255.)
            plt.subplot(1, 3, 2)
            plt.imshow((label - 1)*255.)
            plt.subplot(1, 3, 3)
            plt.imshow(pred_label.cpu().detach().numpy().squeeze()*255.)
            plt.show()

def getLoss(pred_label, target_label):
    p = softmax(pred_label)
    return cross_entropy(p, target_label)

def softmax(input):
    # todo: implement softmax function
    p = torch.exp(input.float()) / torch.sum(torch.exp(input.float()), 1)
    # p1 = F.softmax(input.float(), 1)
    # print("p 0", p[0])
    # print("p1 0", p1[0])
    return p

def cross_entropy(input, targets):
    # todo: implement cross entropy
    # Hint: use the choose function
    # using pad to crop targets
    # input torch.Size([1, 2, 116, 116])
    # targets torch.Size([300, 300])
    delta_x = input.shape[2] - targets.shape[0] 
    delta_y = input.shape[3] - targets.shape[1]
    targets = F.pad(targets, pad=(delta_x // 2, delta_y // 2, delta_x // 2, delta_y // 2), mode='constant', value=0)
    # pred = choose(input, targets)
    # ce = torch.mean(-torch.log(pred))
    ce = F.cross_entropy(input.contiguous().view(-1,2), targets.contiguous().view(-1).long())
    # print("ce: ", ce)
    return ce

# Workaround to use numpy.choose() with PyTorch
def choose(pred_label, true_labels):
    size = pred_label.size()
    ind = np.empty([size[2]*size[3],3], dtype=int)
    i = 0
    for x in range(size[2]):
        for y in range(size[3]):
            ind[i,:] = [true_labels[x,y], x, y]
            i += 1
    pred = pred_label[0,ind[:,0],ind[:,1],ind[:,2]].view(size[2],size[3])
    return pred
    
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int', help='number of epochs')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=2, type='int', help='number of classes')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='data/cells/', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_classes=args.n_classes)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    if WILL_TRAIN:
        train_net(net=net,
            epochs=args.epochs,
            n_classes=args.n_classes,
            gpu=args.gpu,
            data_dir=args.data_dir)

    if WILL_TEST:
        testNet = UNet(n_classes=args.n_classes)
        state_dict = torch.load('data/cells/checkpoints/CP2.pth')
        testNet.load_state_dict(state_dict)
        test_net(testNet=testNet, 
            gpu=args.gpu,
            data_dir=args.data_dir)
