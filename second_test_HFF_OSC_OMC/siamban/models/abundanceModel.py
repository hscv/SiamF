import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AbundanceExtract(nn.Module): # get initial w0
    def __init__(self, R=3):
        super(AbundanceExtract, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=(1,1,1))
        self.batch1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=(1,1,1), stride=(1,2,2))
        self.batch2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU()

        self.adaptivePool = nn.AdaptiveAvgPool2d((None,1))
        self.conv3 = nn.Conv2d(32*R,64,1,1)
        self.weight = nn.Parameter(torch.zeros(1,64,4,4))


    def forward(self, x):  # return 16 bands point-mul result
        fea = self.conv1(x)
        fea = self.batch1(fea)
        fea = self.relu1(fea)
        fea = self.conv2(fea)
        fea = self.batch2(fea)
        fea = self.relu2(fea)
        b,c,a,w,h = fea.size()
        fea = fea.view(b,-1,w,h) #.permute(0,2,3,1) ## b,w,h,32*R
        fea = self.conv3(fea)
        out = F.conv2d(fea, self.weight, bias=None, stride=1, padding=(2,2), dilation=1, groups=1)
        out = out[:,:,:-1,:-1]
        return out

if __name__ == '__main__':
    model = AbundanceExtract()
    data = torch.rand(23,1,8,62,62)*255
    output = model(data)
    print (output.size())

    


