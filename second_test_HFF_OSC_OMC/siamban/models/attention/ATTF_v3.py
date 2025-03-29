import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import init
import functools
from torch.autograd import Variable
class MSCAB(nn.Module):
    def __init__(self,in_channels=256, out_channels=256):
        super(MSCAB, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, out_channels//8, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(out_channels//8),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(out_channels//8, in_channels, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(in_channels))

        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, out_channels//8, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(out_channels//8),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(out_channels//8, in_channels, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(in_channels))
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ReLU(inplace=True)
    def forward(self, fea):
        f1 = F.adaptive_avg_pool2d(fea, (1, 1))
        f1 = self.activation(f1) 
        f1 = self.branch1(f1) 
        f2 = self.branch2(fea)
        f1 = f1.expand_as(f2) 
        f3 = f1+f2
        attention = self.sigmoid(f3)
        return attention


class MSCAB_V2(nn.Module):
    def __init__(self,in_channels=256, out_channels=256):
        super(MSCAB_V2, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, out_channels//8, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(out_channels//8),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(out_channels//8, in_channels, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(in_channels),
                                     nn.Sigmoid())

        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, out_channels//8, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(out_channels//8),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(out_channels//8, in_channels, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(in_channels),
                                     nn.Sigmoid())
        self.activation = nn.ReLU(inplace=True)
    def forward(self, fea):
        f1 = F.adaptive_avg_pool2d(fea, (1, 1))
        f1 = self.activation(f1) 
        attention1 = self.branch1(f1) 
        attention1 = attention1.expand_as(fea) 
        attention2 = self.branch2(fea) 
        return fea*attention1+fea*attention2

def ATTF(featureArr,modelMSCAB,weighted_status=False):
    b,c,w,h = featureArr[0].size()
    if modelMSCAB is None:
        resFea = 0
        for i in range(len(featureArr)):
            resFea += featureArr[i]
        resFea = resFea / len(featureArr)
    else:
        shared_fea = 0 
        for fea in featureArr:
            shared_fea += fea
        shared_fea = shared_fea / len(featureArr)
        shared_fea = modelMSCAB(shared_fea)
        
        resFea = 0
        if weighted_status:
            feaLen = len(featureArr)
            loc_weight = nn.Parameter(torch.ones(feaLen))
            loc_weight = F.softmax(loc_weight, 0)

            for i in range(feaLen):
                tt = loc_weight[i] * shared_fea * featureArr[i]
                resFea += tt
        else:
            if len(featureArr) == 5:
                resFea += shared_fea * featureArr[0]
                feaLen = len(featureArr)-1
                for i in range(feaLen):
                    tt = 1.0/(feaLen)*(1-shared_fea) * featureArr[i+1]
                    resFea += tt
            else:
                feaLen = len(featureArr)-1
                resFea += shared_fea * featureArr[feaLen]
                for i in range(feaLen):
                    tt = 1.0/(feaLen)*(1-shared_fea) * featureArr[i]
                    resFea += tt
    return resFea

if __name__ == '__main__':
    pass
