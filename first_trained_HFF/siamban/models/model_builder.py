# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.loss import select_cross_entropy_loss, select_iou_loss
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head
from siamban.models.neck import get_neck
from siamban.models.attention.ATTF_v3 import MSCAB

class Channel_attention_net(nn.Module):

    def __init__(self, channel=16, reduction=4):
        super(Channel_attention_net, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(channel, channel//2,bias=True),
                                     nn.ReLU(inplace=False),
                                     nn.Linear(channel//2, channel//4,bias=True))

        self.decoder = nn.Sequential(nn.Linear(channel//4, channel//2,bias=True),
                                     nn.ReLU(inplace=False),
                                     nn.Linear(channel//2, channel,bias=True)
                                     )
        self.soft = nn.Softmax(dim=-1)
        self.lambd = nn.Parameter(torch.Tensor([0.0001]))

    def forward(self, x):  
        b, c, w, h = x.size()
        c1 = x.view(b,c,-1)
        c2 = c1.permute(0,2,1)
        res1 = self.encoder(c2)
        res2 = self.decoder(res1)
        res2 = res2 / res2.max() 
        res2 = self.soft(res2)
        res = res2.permute(0,2,1)
        att = res.view(b,c,w,h)
        y = res.mean(dim=2)
        y = y.view(b,c,1)
        ty = y.permute(0,2,1)
        w0 = torch.bmm(y,ty)

        y0 = w0.mean(dim=-1)
        w0 = w0/w0.max()
        orderY = torch.sort(y0, dim=-1, descending=True, out=None)  
        rx = x.view(b,c,-1)
        res = x + (self.lambd*torch.bmm(w0,rx)).view(b,c,w,h)
        return res,w0,orderY


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build ban head
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)
        if True:
            self.chanModel = Channel_attention_net().cuda()
            self.MSCABModel_b1 = MSCAB(256,256).cuda()
            self.MSCABModel_b2 = MSCAB(512,512).cuda()
            self.MSCABModel_b3 = MSCAB(1024,1024).cuda()
            self.MSCABModel_b4 = MSCAB(2048,2048).cuda()
        self.adptiveFusionFeaStatus = False
        if self.adptiveFusionFeaStatus:
            self.adpWeModel = WeFusion(5,weightStatus=True)



    def _split_Channel(self,feat_channel,order):
        res = []
        b = feat_channel.size()[0]
        for i in range(5):
            gg = feat_channel[None,0,order[0,i*3:i*3+3],:,:]
            for k in range(1,b):
                gg = torch.cat((gg,feat_channel[None,k,order[k,i*3:i*3+3],:,:]),dim=0)
            res.append(gg)  
        return res

    def template(self, z):
        falseColor = False
        if falseColor:
            zf = self.backbone(z)
            if cfg.ADJUST.ADJUST:
                zf = self.neck(zf)
            self.zf = zf
        else:
            res, w3, orderY = self.chanModel(z)
            order = orderY[1]
            zArr = self._split_Channel(res,order)
            zf = self.backbone(zArr,self.MSCABModel_b1, self.MSCABModel_b2, self.MSCABModel_b3, self.MSCABModel_b4)
            if cfg.ADJUST.ADJUST:
                zf = self.neck(zf)
                self.zf = zf
            else:
                self.zf = zf
        

    def track(self, x):
        res, w, orderY = self.chanModel(x)
        order = orderY[1]
        xArr = self._split_Channel(res,order)
        xf = self.backbone(xArr,self.MSCABModel_b1, self.MSCABModel_b2, self.MSCABModel_b3, self.MSCABModel_b4)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'order': orderY
               }


    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()

        # get feature
        res_template, w, orderY_template = self.chanModel(template) 
        order_template = orderY_template[1]
        templateArr = self._split_Channel(res_template,order_template)
        
        res_search, w, orderY_search = self.chanModel(search)
        order_search = orderY_search[1] 
        searchArr = self._split_Channel(res_search,order_search)
        penalty = orderY_search[0].sum(dim=0) * 1.0 / order_search.size()[0]
        pe_arr = []
        for k in range(5):
            pe_arr.append(penalty[k*3:(k+1)*3].sum())

        xArr = []
        zArr = []
        for i in range(5):
            z = templateArr[i]
            x = searchArr[i]
            xArr.append(x)
            zArr.append(z)
        zf = self.backbone(zArr,self.MSCABModel_b1, self.MSCABModel_b2, self.MSCABModel_b3, self.MSCABModel_b4)
        xf = self.backbone(xArr,self.MSCABModel_b1, self.MSCABModel_b2, self.MSCABModel_b3, self.MSCABModel_b4)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf) 
            xf = self.neck(xf) 

        cls, loc = self.head(zf, xf) 

        # cls loss with cross entropy loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = select_iou_loss(loc, label_loc, label_cls)
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss 
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        return outputs
