import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.conv1 = downStep(1, 64)
        self.conv2 = downStep(64, 128)
        self.conv3 = downStep(128, 256)
        self.conv4 = downStep(256, 512)
        self.conv5 = downStep(512, 1024)  
        self.down_pooling = nn.MaxPool2d(2)
        self.conv6 = upStep(1024, 512)
        self.conv7 = upStep(512, 256)
        self.conv8 = upStep(256, 128)
        self.conv9 = upStep(128, 64, withReLU=False)
        self.conv10 = nn.Conv2d(64, n_classes, 1)  # last convolutional layer 
        
    def forward(self, x):
        # todo
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)
        x6 = self.conv6(x5,x4)
        x7 = self.conv7(x6,x3)
        x8 = self.conv8(x7,x2)
        x9 = self.conv9(x8,x1)
        x10 = self.conv10(x9)
        x = torch.sigmoid(x10)
        return x

class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        # todo
        self.downConv = nn.Sequential(
            nn.Conv2d(inC, outC, 3),
            nn.BatchNorm2d(outC), 
            nn.ReLU(inplace=True),
            nn.Conv2d(outC, outC, 3),
            nn.BatchNorm2d(outC),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # todo  
        x = self.downConv(x)   
        return x

class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        # todo
        # Do not forget to concatenate with respective step in contracting path
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!        
        self.uppooling = nn.ConvTranspose2d(inC, outC, 2, stride=2)
        if(withReLU):
            self.upConv = nn.Sequential(
                nn.Conv2d(inC, outC, 3),
                nn.BatchNorm2d(outC), 
                nn.ReLU(inplace=True),
                nn.Conv2d(outC, outC, 3),
                nn.BatchNorm2d(outC),
                nn.ReLU(inplace=True),
            )
        else:
            self.upConv = nn.Sequential(
                nn.Conv2d(inC, outC, 3),
                nn.BatchNorm2d(outC), 
                nn.Conv2d(outC, outC, 3),
                nn.BatchNorm2d(outC),
            )

    def forward(self, x, x_down):
        # todo
        x = self.uppooling(x)
        pad_y = x.size()[2] - x_down.size()[2]
        pad_x = x.size()[3] - x_down.size()[3]
        x_down = F.pad(x_down, pad=(pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2))
        x = torch.cat([x_down, x], dim=1)
        x = self.upConv(x)
        return x

