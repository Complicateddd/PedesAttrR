import math
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from itertools import chain
from torch.nn import functional as F
# from backbone.resnet import conv3x3

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BaseClassifier(nn.Module):
    def __init__(self, nattr):
        super().__init__()
        self.logits = nn.Sequential(
            nn.Linear(2048, nattr),
            nn.BatchNorm1d(nattr)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def fresh_params(self):
        return self.parameters()

    def forward(self, feature):
        feat = self.avg_pool(feature).view(feature.size(0), -1)
        x = self.logits(feat)
        return x

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


class MSSC(nn.Module):

    def __init__(self, backbone, classifier):
        super(MSSC, self).__init__()
        inplace = True
        self.backbone = backbone
        self.classifier = classifier

        self.scc_1 = SCM(in_channels=256)
        self.scc_2 = SCM(in_channels=256)
        self.scc_3 = SCM(in_channels=256)

        # self.msff_1 = MSFF(in_channels=512)
        self.msff_2 = MSFF(in_channels=512)
        self.msff_3 = MSFF(in_channels=512)
        self.msff_4 = MSFF(in_channels=256)

        # self.sap_1 = SingleAttributePrediction(num_classes=21, channels=512)
        self.sap_2 = SingleAttributePrediction(num_classes=21, channels=512)
        self.sap_3 = SingleAttributePrediction(num_classes=21, channels=512)
        self.sap_4 = SingleAttributePrediction(num_classes=21, channels=256)

        # self.latlayer_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.latlayer_2 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.latlayer_3 = nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1)
        self.latlayer_4 = nn.Conv2d(1536, 256, kernel_size=3, stride=1, padding=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        up_feat = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
        return torch.cat([up_feat, y], 1)


    def fresh_params(self):
        params = chain(self.classifier.fresh_params(True), self.sap_2.parameters(), self.sap_3.parameters(), self.sap_4.parameters(),
                       self.latlayer_2.parameters(), self.latlayer_3.parameters(), self.latlayer_4.parameters(),
                       self.msff_2.parameters(),self.msff_3.parameters(), self.msff_4.parameters(),
                       self.scc_2.parameters(), self.scc_3.parameters())
        return params

    def finetune_params(self):
        return self.backbone.parameters()


    def forward(self, x, label=None):
        feat_layer2, feat_layer3, feat_layer4 = self.backbone(x)
        logits,_ = self.classifier(feat_layer4)
        logits = logits[0]

        fusion_4 = self.latlayer_4(feat_layer4)
        fusion_3 = self.scc_3(fusion_4, self.latlayer_3(feat_layer3))
        fusion_2 = self.scc_2(fusion_3, self.latlayer_2(feat_layer2))
        # fusion_1 = self.scc_1(fusion_2, self.latlayer_1(feat_layer1))

        fusion_3 = self.msff_3(fusion_3)
        fusion_2 = self.msff_2(fusion_2)
        # fusion_1 = self.msff_1(fusion_1)
        fusion_4 = self.msff_4(fusion_4)

        # pred_1 = self.sap_1(fusion_1)
        pred_2 = self.sap_2(fusion_2)
        pred_3 = self.sap_3(fusion_3)
        pred_4 = self.sap_4(fusion_4)

        return [pred_2, pred_3, pred_4, logits],None


class SCM(nn.Module):
    def __init__(self, in_channels, stride=1, spatial_up="end"):
        super(SCM, self).__init__()
        self.spatial = spatial_up
        self.k3 = nn.Sequential(
            conv3x3(in_channels, in_channels),
            nn.BatchNorm2d(in_channels))

        self.k4 = nn.Sequential(
            conv3x3(in_channels , in_channels, stride),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.k1 = nn.Sequential(
            conv3x3(in_channels, in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self._split = nn.Sequential(
            conv3x3(in_channels * 2, in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

    def forward(self, x, y):

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if x.size()[1] != 256:  # Control the channel of x to 256
            x = self._split(x)

        identity = y

        # Interpolate determines the size of up-sample
        out = torch.sigmoid(torch.add(identity, x))
        out = torch.mul(self.k3(y), out)  # k3 * sigmoid(identity + k2)
        y1 = self.k4(out)  # k4
        y2 = self.k1(y)
        output = torch.cat([y1, y2], dim=1)
        return output


class ChannelAttn(nn.Module):
    """Constructs a Eca Attention module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ChannelAttn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


# Single attribute prediction
class SingleAttributePrediction(nn.Module):
    def __init__(self, num_classes, channels):
        super(SingleAttributePrediction, self).__init__()
        self.num_classes = num_classes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.gap_list = nn.ModuleList()
        self.fc_list = nn.ModuleList()
        self.att_list = nn.ModuleList()
        for i in range(self.num_classes):
            self.gap_list.append(nn.AdaptiveAvgPool2d(1))
            self.fc_list.append(nn.Linear(channels, 1))
            self.att_list.append(ChannelAttn(channels))

    def forward(self, features):
        pred_list = []
        bs = features.size(0)
        for i in range(self.num_classes):
            att_feature = self.att_list[i](features) + features
            pred = self.gap_list[i](att_feature).view(bs,-1)
            pred = self.fc_list[i](pred)
            pred_list.append(pred)
        pred = torch.cat(pred_list, 1)
        return pred



class MSFF(nn.Module):
    """
    MSFF(in_channels=32),
    super(NONLocalBlock2D, self).__init__(in_channels,
            inter_channels=inter_channels,
            dimension=2, sub_sample=sub_sample,
            bn_layer=bn_layer)
    """

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=2,
                 sub_sample=True,
                 bn_layer=True):
        super(MSFF, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # Compress to get the number of channels
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c,  h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x   # W_y: [16, 256, 8, 6]
        return z


# net = MSSC(None,None)


