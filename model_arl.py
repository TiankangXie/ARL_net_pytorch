import torch.nn as nn
import torch
from crfrnn import CrfRnn


class LocalConv2dReLU(nn.Module):
    """
    The output feature is divided into 8x8 patches (or 4x4 or 2x2).
    For each of the patch we designed a set of convolutional layer, with batch normalization and RELU, 
    Thus we will have 64 sets for 8x8 patch layer, and 16 sets for 4x4 patch layer.
    """

    def __init__(self, local_h_num, local_w_num, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, activation_type='ReLu'):
        super(LocalConv2dReLU, self).__init__()
        self.local_h_num = local_h_num  # number of patches for height
        self.local_w_num = local_w_num  # number of patches for width

        self.bns = nn.ModuleList([nn.BatchNorm2d(in_channels)
                                  for i in range(local_h_num*local_w_num)])

        if activation_type == 'ReLU':
            self.relus = nn.ModuleList(
                [nn.ReLU(inplace=True) for i in range(local_h_num*local_w_num)])
        elif activation_type == 'PReLU':
            self.relus = nn.ModuleList([nn.PReLU()
                                        for i in range(local_h_num*local_w_num)])

        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              bias) for i in range(local_h_num * local_w_num)])

    def forward(self, x):
        """
        To forward our model, we need to first divide our tensor (picture) into patches.
        Then we use the sets of patch layers to process each parcel.
        """
        h_splits = torch.split(x, int(x.size(2) / self.local_h_num), dim=2)
        h_out = []

        for i in range(len(h_splits)):
            start = True  # Just to see if it is the first layer in the patches
            w_splits = torch.split(h_splits[i], int(
                h_splits[i].size(3)/self.local_w_num), dim=3)
            for j in range(len(w_splits)):
                bn_out = self.bns[i*len(w_splits)+j](w_splits[j].contiguous())
                bn_out = self.relus[i*len(w_splits)+j](bn_out)
                conv_out = self.convs[i*len(w_splits)+j](bn_out)
                if start:
                    h_out.append(conv_out)
                    start = False
                else:
                    h_out[i] = torch.cat((h_out[i], conv_out), 3)
            if i == 0:
                out = h_out[i]
            else:
                out = torch.cat((out, h_out[i]), 2)
        return(out)


class HierarchicalMultiScaleRegionLayer(nn.Module):
    """
    Employs the modules from LocalConv2dReLU to perform patching on our dataset
    """

    def __init__(self, local_group, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, activation_type='ReLU'):
        super(HierarchicalMultiScaleRegionLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)

        self.local_conv_branch1 = LocalConv2dReLU(local_group[0][0], local_group[0][1], out_channels, int(out_channels / 2),
                                                  kernel_size, stride,
                                                  padding, dilation, groups, bias, activation_type)
        self.local_conv_branch2 = LocalConv2dReLU(local_group[1][0], local_group[1][1], int(out_channels / 2),
                                                  int(out_channels /
                                                      4), kernel_size, stride,
                                                  padding, dilation, groups, bias, activation_type)
        self.local_conv_branch3 = LocalConv2dReLU(local_group[2][0], local_group[2][1], int(out_channels / 4),
                                                  int(out_channels /
                                                      4), kernel_size, stride,
                                                  padding, dilation, groups, bias, activation_type)

        # Not sure why the feature shape is out_channels.
        self.bn = nn.BatchNorm2d(out_channels)

        if activation_type == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif activation_type == 'PReLU':
            self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        local_branch1 = self.local_conv_branch1(x)
        local_branch2 = self.local_conv_branch2(local_branch1)
        local_branch3 = self.local_conv_branch3(local_branch2)
        local_out = torch.cat((local_branch1, local_branch2, local_branch3), 1)

        out = x + local_out
        out = self.bn(out)
        out = self.relu(out)

        return(out)


class HMRegionLearning(nn.Module):
    def __init__(self, input_dim=3, unit_dim=8):
        super(HMRegionLearning, self).__init__()

        self.multiscale_feat = nn.Sequential(
            HierarchicalMultiScaleRegionLayer([[8, 8], [4, 4], [2, 2]], input_dim, unit_dim * 4, kernel_size=3,
                                              stride=1, padding=1,
                                              activation_type='ReLU'),
            nn.MaxPool2d(kernel_size=2, stride=2),

            HierarchicalMultiScaleRegionLayer([[8, 8], [4, 4], [2, 2]], unit_dim * 4, unit_dim * 8, kernel_size=3,
                                              stride=1, padding=1,
                                              activation_type='ReLU'),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        multiscale_feat = self.multiscale_feat(x)
        return (multiscale_feat)


class ChannelWiseSpatialAttentLearning(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, num_iters=5):
        super(ChannelWiseSpatialAttentLearning, self).__init__()

        self.conv_layer0 = nn.ModuleList([nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias) for i in range(5)])

        self.conv_layer1 = nn.Conv2d(
            in_channels, int(
                in_channels/4), kernel_size, stride, padding, dilation, groups, bias
        )
        self.conv_layer2 = nn.Conv2d(
            int(in_channels/4), 1, kernel_size, stride, padding, dilation, groups, bias
        )

        self.full_connected_layer = nn.Sequential(nn.Linear(in_channels, in_channels, bias=False),
                                                  nn.Sigmoid(),
                                                  )
        self.full_connected_layer2 = nn.Sequential(nn.Linear(in_channels, 1, bias=True),
                                                   nn.Sigmoid()
                                                   )

        self.crf_nn = CrfRnn(num_labels=2, num_iterations=num_iters)

        self.relus = nn.ModuleList(
            [nn.ReLU(inplace=True) for i in range(6)])

        self.bns = nn.ModuleList([nn.BatchNorm2d(in_channels)
                                  for i in range(local_h_num*local_w_num)])

    def forward(self, x):

        f_1 = self.conv_layer0[0](x)
        f_1 = self.relus[0](f_1)
        f_1c = nn.AvgPool2d(
            kernel_size=x.shape[2], stride=0, padding=0, ceil_mode=False, count_include_pad=True)(f_1)

        f_2 = self.conv_layer0[1](f_1c)
        f_2 = self.relus[1](f_2)

        v_c = self.full_connected_layer(f_1c)

        f_c = v_c * f_2
        f_3 = self.conv_layer0[2](f_c)
        f_3 = self.relus[2](f_3)
        f_4 = self.conv_layer0[3](f_3)
        f_4 = self.relus[3](f_4)

        f_3s = self.conv_layer1(f_3)
        f_3s = self.relus[4](f_3s)

        v_0s = self.conv_layer2(f_3s)
        v_0s = self.relus[5](v_0s)

        v_s = self.crf_nn(f_3s, v_0s)

        f_s = v_s * f_4

        f_r_before = self.conv_layer0[4](f_s)
        f_r_before = self.relus[6](f_r_before)

        f_r_after = nn.AvgPool2d(
            kernel_size=f_r_before.shape[-1], stride=0, padding=0, ceil_mode=False, count_include_pad=True)(f_r_before)

        p_n = self.full_connected_layer2(f_r_after)

        return(p_n)
