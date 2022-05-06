import math

import torch
import torch.nn as nn
import torch.nn.init as init

import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from models.registry import CLASSIFIER

from models.csra import MHA

from models.alm import ChannelAttn,SpatialTransformBlock
from itertools import chain

class BaseClassifier(nn.Module):

    def fresh_params(self, bn_wd):
        if bn_wd:
            return self.parameters()
        else:
            return self.named_parameters()

@CLASSIFIER.register("linear")
class LinearClassifier(BaseClassifier):
    def __init__(self, nattr, c_in, bn=False, pool='avg', scale=1):
        super().__init__()

        self.pool = pool
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)

        self.logits = nn.Sequential(
            nn.Linear(c_in, nattr),
            nn.BatchNorm1d(nattr) if bn else nn.Identity()
        )


    def forward(self, feature, label=None):

        if len(feature.shape) == 3:  # for vit (bt, nattr, c)

            bt, hw, c = feature.shape
            # NOTE ONLY USED FOR INPUT SIZE (256, 192)
            h = 16
            w = 12
            feature = feature.reshape(bt, h, w, c).permute(0, 3, 1, 2)

        feat = self.pool(feature).view(feature.size(0), -1)
        x = self.logits(feat)

        return [x], feature



@CLASSIFIER.register("cosine")
class NormClassifier(BaseClassifier):
    def __init__(self, nattr, c_in, bn=False, pool='avg', scale=30):
        super().__init__()

        self.logits = nn.Parameter(torch.FloatTensor(nattr, c_in))

        stdv = 1. / math.sqrt(self.logits.data.size(1))
        self.logits.data.uniform_(-stdv, stdv)

        self.pool = pool
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, feature, label=None):
        feat = self.pool(feature).view(feature.size(0), -1)
        feat_n = F.normalize(feat, dim=1)
        weight_n = F.normalize(self.logits, dim=1)
        x = torch.matmul(feat_n, weight_n.t())
        return [x], feat_n


################ CSRA Classifier ####################
@CLASSIFIER.register("csra")
class CSRAClassifier(BaseClassifier):
    def __init__(self, nattr, c_in, bn=False, pool='avg', scale=30):
        super().__init__()

        self.logits = MHA(num_heads=4, lam = 0.2, input_dim=c_in, num_classes=nattr)

    def forward(self, feature, label=None):
        
        x = self.logits(feature)

        return [x],None


################ CSRA Classifier ####################
# @CLASSIFIER.register("csra_fpn")
# class CSRAClassifierFPN(BaseClassifier):
#     def __init__(self,nattr, c_in, bn=False, pool='avg', scale=30):
#         super().__init__()


#         self.letten1 = nn.Sequential(nn.Conv2d(1024, 1024,1,1,0),nn.BatchNorm2d(1024))
#         self.letten2 = nn.Sequential(nn.Conv2d(1024, 1024,1,1,0),nn.BatchNorm2d(1024))
#         self.letten3 = nn.Sequential(nn.Conv2d(1024, 1024,1,1,0),nn.BatchNorm2d(1024))
#         self.letten4 = nn.Sequential(nn.Conv2d(1024, 1024,1,1,0),nn.BatchNorm2d(1024))

#         self.logits1 = MHA(num_heads=4, lam = 0.3, input_dim=1024, num_classes=4)
#         self.logits2 = MHA(num_heads=4, lam = 0.3, input_dim=1024, num_classes=3)
#         self.logits3 = MHA(num_heads=4, lam = 0.3, input_dim=1024, num_classes=3)
#         self.logits4 = MHA(num_heads=4, lam = 0.3, input_dim=1024, num_classes=11)


#     def forward(self, feature, label=None):

#         # print(feature.shape)

#         pred_11 = self.letten1(feature)
#         pred_21 = self.letten2(feature)
#         pred_31 = self.letten3(feature)
#         pred_41 = self.letten4(feature)

#         pred_1 = self.logits1(pred_11)
#         pred_2 = self.logits2(pred_21)
#         pred_3 = self.logits3(pred_31)
#         pred_4 = self.logits4(pred_41)

#         return [pred_1,pred_2,pred_3,pred_4],None

@CLASSIFIER.register("csra_fpn")
class CSRAClassifierFPN(BaseClassifier):
    def __init__(self,nattr, c_in, bn=False, pool='avg', scale=30):
        super().__init__()


        self.letten1 = nn.Sequential(nn.Conv2d(1024, 1024,1,1,0),nn.BatchNorm2d(1024))
        self.letten2 = nn.Sequential(nn.Conv2d(1024, 1024,1,1,0),nn.BatchNorm2d(1024))
        self.letten3 = nn.Sequential(nn.Conv2d(1024, 1024,1,1,0),nn.BatchNorm2d(1024))
        self.letten4 = nn.Sequential(nn.Conv2d(1024, 1024,1,1,0),nn.BatchNorm2d(1024))

        self.logits1 = MHA(num_heads=4, lam = 0.3, input_dim=1024, num_classes=21)
        self.logits2 = MHA(num_heads=4, lam = 0.3, input_dim=1024, num_classes=21)
        self.logits3 = MHA(num_heads=4, lam = 0.3, input_dim=1024, num_classes=21)
        self.logits4 = MHA(num_heads=4, lam = 0.3, input_dim=1024, num_classes=21)


    def forward(self, feature, label=None):

        # print(feature.shape)
        p2,p3,p4 = feature

        pred_11 = self.letten1(p2)
        pred_21 = self.letten2(p4)
        pred_31 = self.letten3(p3)
        pred_41 = self.letten4(p3)

        pred_1 = self.logits1(pred_11)
        pred_2 = self.logits2(pred_21)
        pred_3 = self.logits3(pred_31)
        pred_4 = self.logits4(pred_41)

        return [pred_1,pred_2,pred_3,pred_4],None






def initialize_weights(module):
    for m in module.children():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, _BatchNorm):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)


class FeatClassifier(nn.Module):

    def __init__(self, backbone, classifier, bn_wd=True):
        super(FeatClassifier, self).__init__()

        self.backbone = backbone
        self.classifier = classifier
        self.bn_wd = bn_wd

    def fresh_params(self):
        return self.classifier.fresh_params(self.bn_wd)

    def finetune_params(self):

        if self.bn_wd:
            return self.backbone.parameters()
        else:
            return self.backbone.named_parameters()

    def forward(self, x, label=None):
        feat_map = self.backbone(x)
        logits, feat = self.classifier(feat_map, label)
        return logits, feat



class SwinALM(nn.Module):
    def __init__(self, fpn_backbone,classifier,num_classes=21,bn_wd=True):
        super(SwinALM, self).__init__()
        self.num_classes = num_classes
        self.fpn_backbone = fpn_backbone
        self.bn_wd = bn_wd

        self.classifier = classifier

        self.st_3b = SpatialTransformBlock(num_classes, 32, 512*3)
        self.st_4d = SpatialTransformBlock(num_classes, 16, 512*2)
        self.st_5b = SpatialTransformBlock(num_classes, 8, 512)

        # Lateral layers
        # self.latlayer_3b = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer_4d = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer_5b = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)

        self.latlayer_3b = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer_4d = nn.Conv2d(768, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer_5b = nn.Conv2d(1536, 512, kernel_size=1, stride=1, padding=0)


    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        up_feat = F.interpolate(x, (H, W), mode='bilinear', align_corners=False)
        return torch.cat([up_feat,y], 1)

    def forward(self, x,label=None):
        bs = x.size(0)

        feat_3b, feat_4d, feat_5b = self.fpn_backbone(x)

        
        # main_pred = None;
        # print(feat_5b)
        main_pred,_ = self.classifier(feat_5b)
        main_pred = main_pred[0]

        fusion_5b = self.latlayer_5b(feat_5b)
        fusion_4d = self._upsample_add(fusion_5b, self.latlayer_4d(feat_4d))
        fusion_3b = self._upsample_add(fusion_4d, self.latlayer_3b(feat_3b))

        # print(fusion_3b.shape,fusion_4d.shape,fusion_3bn_5b.shape,main_pred.shape)
        # print(fusion_3b.shape)
        pred_3b = self.st_3b(fusion_3b)
        pred_4d = self.st_4d(fusion_4d)
        pred_5b = self.st_5b(fusion_5b)

        return [pred_3b, pred_4d, pred_5b, main_pred],None

    def fresh_params(self):
        return chain(
            self.classifier.fresh_params(self.bn_wd),
            self.st_3b.parameters(),self.st_4d.parameters(),self.st_5b.parameters(),
            self.latlayer_3b.parameters(),self.latlayer_4d.parameters(),self.latlayer_5b.parameters()
            )

    def finetune_params(self):

        if self.bn_wd:
            return self.fpn_backbone.parameters()
        else:
            return self.fpn_backbone.named_parameters()


class FPN_Neck(nn.Module):
    def __init__(self):
        super().__init__()

        self.top_layer = nn.Conv2d(1024, 1024, 1, 1, 0)

        self.lateral_1 = nn.Conv2d(512, 1024, 1, 1, 0)
        self.lateral_2 = nn.Conv2d(256, 1024, 1, 1, 0)

        self.smooth_1 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        self.smooth_2 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        self.smooth_3 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)

    def forward(self, c3,c4,c5):

        p4 = self.top_layer(c5)

        p3 = self.upscale_add(p4, self.lateral_1(c4))
        p2 = self.upscale_add(p3, self.lateral_2(c3))

        p3 = self.smooth_1(p3)
        p2 = self.smooth_2(p2)

        # global_part = p4

        # up_part_1 = p2[:,:,:12,:]
        # up_part_2 = p3[:,:,:6,:]

        # middle_part_1 = p2[:,:,8:24,:]
        # middle_part_2 = p3[:,:,4:12,:]

        # return [up_part_1, up_part_2,middle_part_1, middle_part_2,global_part]
        return [p2,p3,p4]

    def upscale_add(self, x, y):
        _, _, H, W = y.size()
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear',align_corners=False) + y


class PoolingSwin(nn.Module):
    def __init__(self, fpn_backbone,classifier,num_classes=21,bn_wd=True):
        super(PoolingSwin, self).__init__()
        self.num_classes = num_classes
        self.fpn_backbone = fpn_backbone
        self.bn_wd = bn_wd

        self.classifier = classifier

        self.maxpooling = nn.MaxPool2d(2,2)
        
        self.Conv3 = nn.Conv2d(256, 512, 1 , 1, 0)
        self.Bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(inplace=True)

        self.Conv4 = nn.Conv2d(512, 1024, 1 , 1, 0)
        self.Bn4 = nn.BatchNorm2d(1024)
        self.relu4 = nn.ReLU(inplace=True)


    def forward(self, x,label=None):
        bs = x.size(0)

        feat_3b, feat_4b, feat_5b = self.fpn_backbone(x)

        feat_3b = self.Conv3(feat_3b)
        feat_3b = self.Bn3(feat_3b)
        feat_3b = self.relu3(feat_3b)
        feat_3b = self.maxpooling(feat_3b) 

        feat_4b = feat_4b +feat_3b

        feat_4b = self.Conv4(feat_4b)
        feat_4b = self.Bn4(feat_4b)
        feat_4b = self.relu4(feat_4b)
        feat_4b = self.maxpooling(feat_4b) 

        feat_final = torch.cat((feat_4b,feat_5b),1)

        main_pred,_ = self.classifier(feat_final)

        return main_pred,None

class PoolingSwin2(nn.Module):
    def __init__(self, fpn_backbone,classifier,num_classes=21,bn_wd=True):
        super(PoolingSwin2, self).__init__()
        self.num_classes = num_classes
        self.fpn_backbone = fpn_backbone
        self.bn_wd = bn_wd

        self.classifier = classifier

        self.maxpooling = nn.MaxPool2d(2,2)
        
        self.Conv3 = nn.Conv2d(384, 768, 3 , 2, 1)
        self.Bn3 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)

        self.Conv4 = nn.Conv2d(768, 1536, 3 , 2, 1)
        self.Bn4 = nn.BatchNorm2d(1536)
        self.relu4 = nn.ReLU(inplace=True)

        self.Conv3_l = nn.Conv2d(768, 1536, 3 , 2, 1)
        self.Bn3_l = nn.BatchNorm2d(1536)
        self.relu3_l = nn.ReLU(inplace=True)


    def forward(self, x,label=None):
        bs = x.size(0)

        feat_3b, feat_4b, feat_5b = self.fpn_backbone(x)

        feat_3b = self.Conv3(feat_3b)
        feat_3b = self.Bn3(feat_3b)
        feat_3b = self.relu3(feat_3b)

        # feat_3b = self.maxpooling(feat_3b) 

        pred_3b = self.Conv3_l(feat_3b)
        pred_3b = self.Bn3_l(pred_3b)
        pred_3b = self.relu3_l(pred_3b)
        # pred_3b = self.maxpooling(pred_3b) 

        feat_4b = feat_4b + feat_3b

        feat_4b = self.Conv4(feat_4b)
        feat_4b = self.Bn4(feat_4b)
        feat_4b = self.relu4(feat_4b)
        # feat_4b = self.maxpooling(feat_4b) 
        # print(pred_3b.shape,feat_4b.shape,feat_5b.shape)
        # feat_final = torch.cat((pred_3b,feat_4b,feat_5b),1)
        feat_final = pred_3b+feat_4b+feat_5b

        main_pred,_ = self.classifier(feat_final)

        return main_pred,None

    def fresh_params(self):
        return self.classifier.fresh_params(self.bn_wd)

    def finetune_params(self):

        if self.bn_wd:
            return self.fpn_backbone.parameters()
        else:
            return self.fpn_backbone.named_parameters()


class FPNNeckSwin(nn.Module):
    def __init__(self, fpn_backbone,fpn_classifier,num_classes=21,bn_wd=True):
        super(FPNNeckSwin, self).__init__()

        self.bn_wd = bn_wd
        self.fpn_backbone = fpn_backbone
        self.neck = FPN_Neck()
        self.fpn_classifier = fpn_classifier


    def forward(self, x,label=None):
        bs = x.size(0)

        feat_3b, feat_4b, feat_5b = self.fpn_backbone(x)

        feat_fpn = self.neck(feat_3b, feat_4b, feat_5b)

        main_pred,_ = self.fpn_classifier(feat_fpn)

        return main_pred,None

    def fresh_params(self):
        return self.fpn_classifier.fresh_params(self.bn_wd)

    def finetune_params(self):

        if self.bn_wd:
            return self.fpn_backbone.parameters()
        else:
            return self.fpn_backbone.named_parameters()



# import torch
# import torch.nn as nn
import math


class SSCA(nn.Module):
    def __init__(self,backbone,num_classes=21,bn_wd=True,
        head_num=3,channel=1024,attr=21,batch_size=2):

        super(SSCA,self).__init__()
        self.Q = nn.Parameter(torch.randn(batch_size//2,channel,attr,requires_grad=True))

        self.head_num = head_num
        self.fc = nn.Linear(channel,1)

        self.softmax = nn.Softmax(dim=1)

        self.linear_1 = nn.Conv2d(channel,channel,1,1,0)
        self.linear_2 = nn.Conv2d(channel,channel,1,1,0)

        self.linear_3 = nn.Linear(attr*channel,attr*channel,bias=False)

        self.attr = attr
        self.channel = channel

        self.bn_wd = bn_wd
        self.backbone = backbone

    def ssca_forward(self,Q,feature):
        b,c,h,w = feature.shape
        linear_feature_1 = self.linear_1(feature)
        linear_feature_2 = self.linear_2(feature)
        linear_q = self.linear_3(self.Q.view(b,-1)).reshape(b,c,-1)
        Q =linear_q.permute(0,2,1)
        linear_feature_1 = linear_feature_1.view(b,c,-1)
        linear_feature_2 = linear_feature_2.view(b,c,-1)
        QK = torch.einsum('bij,bjk->bik', Q, linear_feature_1)/math.sqrt(c)
        A = self.softmax(QK)
        final_feature = torch.einsum('bij,bcj->bic',A,linear_feature_2)
        return final_feature

    def forward(self,x,label=None):
        feature = self.backbone(x)
        print(feature.shape)
        Q = self.Q
        for i in range(self.head_num):
            Q = self.ssca_forward(Q,feature)
        pred = self.fc(Q).squeeze(-1)
        return [pred],None

    def fresh_params(self):
        return self.parameters()

    def finetune_params(self):

        if self.bn_wd:
            return self.backbone.parameters()
        else:
            return self.backbone.named_parameters()