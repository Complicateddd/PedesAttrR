

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttn(nn.Module):
    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels%reduction_rate == 0
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_rate, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels // reduction_rate, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return torch.sigmoid(x)


class SpatialTransformBlock(nn.Module):
    def __init__(self, num_classes, pooling_size, channels):
        super(SpatialTransformBlock, self).__init__()
        self.num_classes = num_classes
        self.spatial = pooling_size

        # self.global_pool = nn.AvgPool2d((pooling_size, pooling_size//2), stride=1, padding=0, ceil_mode=True, count_include_pad=True)

        self.gap_list = nn.ModuleList()
        self.fc_list = nn.ModuleList()
        self.att_list = nn.ModuleList()
        self.stn_list = nn.ModuleList()
        for i in range(self.num_classes):
            self.gap_list.append(nn.AvgPool2d((pooling_size, pooling_size*3//4), stride=1, padding=0, ceil_mode=True, count_include_pad=True))
            self.fc_list.append(nn.Linear(channels, 1))
            self.att_list.append(ChannelAttn(channels))
            self.stn_list.append(nn.Linear(channels, 4))

    def stn(self, x, theta):
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, padding_mode='border')
        return x.cuda()

    def transform_theta(self, theta_i, region_idx):
        theta = torch.zeros(theta_i.size(0), 2, 3)
        theta[:,0,0] = torch.sigmoid(theta_i[:,0])
        theta[:,1,1] = torch.sigmoid(theta_i[:,1])
        theta[:,0,2] = torch.tanh(theta_i[:,2])
        theta[:,1,2] = torch.tanh(theta_i[:,3])
        theta = theta.cuda()
        return theta

    def forward(self, features):
        pred_list = []
        bs = features.size(0)

        # print('##########################',features.shape)

        for i in range(self.num_classes):
            stn_feature = features * self.att_list[i](features) + features

            theta_i = self.stn_list[i](F.avg_pool2d(stn_feature, stn_feature.size()[2:]).view(bs,-1)).view(-1,4)
            theta_i = self.transform_theta(theta_i, i)

            sub_feature = self.stn(stn_feature, theta_i)

            # print('##########################',sub_feature.shape)
            pred = self.gap_list[i](sub_feature).view(bs,-1)
            # print('##########################',pred.shape)

            pred = self.fc_list[i](pred)
            pred_list.append(pred)
        pred = torch.cat(pred_list, 1)
        return pred