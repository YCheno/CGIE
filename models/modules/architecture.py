from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        return torch.cat(x_list, dim=1)


class SPSRNet(nn.Module):
    def __init__(
        self,
        in_nc,
        out_nc,
        nf,
        nb,
        gc=32,
        upscale=4,
        norm_type=None,
        act_type='leakyrelu',
        mode='CNA',
        upsample_mode='upconv',
    ):
        super(SPSRNet, self).__init__()

        self.conv_1 = conv_layer(in_nc, nf, kernel_size=3)
        self.body = nn.Sequential()
        for i in range(nb):
            self.body.add_module('conv_{}'.format(i + 1), Highfrequency_Reconstruction(nf, gc, nf))

        self.maxpooling = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.body2 = nn.Sequential()
        for i in range(nb):
            self.body2.add_module('conv_{}'.format(i + 1), Samplespace_Reconstruction(nf, gc, nf))

        self.shuffle = pixelshuffle_block(nf, nf, upscale_factor=2)
        self.esa = ESA(nf // 2, nf, nn.Conv2d)
        self.conv_2 = conv_layer(2 * nf, nf, kernel_size=3)
        self.get_grad = Get_gradient_nopadding()
        self.body3 = nn.Sequential()
        for i in range(1):
            self.body3.add_module('conv_{}'.format(i + 1), Highfrequency_Reconstruction(nf, gc, nf))
        self.conv_4 = conv_layer(nf, out_nc, kernel_size=3)

    def forward(self, x):
        out_feature1 = self.conv_1(x)
        grad = self.get_grad(out_feature1)
        x_input = out_feature1 - grad

        out_feature = self.body(out_feature1)

        out_feature1 = self.maxpooling(out_feature1)
        out_feature1 = self.body2(out_feature1)
        out_feature1 = self.shuffle(out_feature1)
        out_feature1 = self.esa(out_feature1)

        out_feature = torch.cat([out_feature, out_feature1], dim=1)
        out_feature = self.conv_2(out_feature)
        out_feature = self.body3(out_feature)
        return self.conv_4(out_feature + x_input)


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    """Create a convolution layer with adaptive same padding."""
    kernel_size = _make_pair(kernel_size)
    padding = (
        int((kernel_size[0] - 1) / 2),
        int((kernel_size[1] - 1) / 2),
    )
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=padding,
        bias=bias,
    )


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]

    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    """Upsample features according to `upscale_factor`."""
    conv = conv_layer(
        in_channels,
        out_channels * (upscale_factor ** 2),
        kernel_size,
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), proposed by
    Residual Feature Aggregation Network for Image Super-Resolution.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class Highfrequency_Reconstruction(nn.Module):
    """Residual local feature block for high-frequency reconstruction."""

    def __init__(self, in_channels, mid_channels=None, out_channels=None):
        super(Highfrequency_Reconstruction, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.sigmoid = nn.Sigmoid()
        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)
        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        grad = self.sigmoid(100 * (x - mean))

        out = self.c1_r(x * grad)
        out = self.act(out)
        out = self.c2_r(out)
        out = self.act(out)
        out = self.c3_r(out)
        out = self.act(out)
        out = out + x
        return self.c5(out)


class Samplespace_Reconstruction(nn.Module):
    """Residual local feature block for sample-space reconstruction."""

    def __init__(self, in_channels, mid_channels=None, out_channels=None):
        super(Samplespace_Reconstruction, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)
        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out = self.c1_r(x)
        out = self.act(out)
        out = self.c2_r(out)
        out = self.act(out)
        out = self.c3_r(out)
        out = self.act(out)
        out = out + x
        return self.c5(out)
