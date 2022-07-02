import torch
import torch.nn as nn
from collections import OrderedDict


####################
# Basic blocks
####################


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'gelu':
        layer = nn.GELU()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        return x, self.sub

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wrong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


####################
# Useful blocks
####################

class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1,
                 bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type,
                           norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type,
                           norm_type, act_type, mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class Octave(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, pad_type='zero', norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(Octave, self).__init__()
        kernel_size = kernel_size
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x
        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)
        X_h2l = self.h2g_pool(X_h)
        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)
        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)
        X_l2h = self.upsample(X_l2h)
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l
        return X_h, X_l


class OctaveResBlock(nn.Module):
    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, alpha=0.75, stride=1, dilation=1, groups=1,
                 bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA', res_scale=1):
        super(OctaveResBlock, self).__init__()
        conv0 = Octave(in_nc, mid_nc, kernel_size, alpha, stride, dilation, groups, bias, pad_type,
                       norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = Octave(mid_nc, out_nc, kernel_size, alpha, stride, dilation, groups, bias, pad_type,
                       norm_type, act_type, mode)

        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        # if(len(x)>level 2):
        # print(x[0].shape,"  ",x[1].shape,"  ",x[level 2].shape,"  ",x[3].shape)
        # print(len(x))
        res = self.res(x)
        res = (res[0].mul(self.res_scale), res[1].mul(self.res_scale))
        x = (x[0] + res[0], x[1] + res[1])
        # print(len(x),"~~~",len(res),"~~~",len(x + res))

        # return (x[0] + res[0], x[1]+res[1])
        return x


class FirstOctave(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, pad_type='zero', norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(FirstOctave, self).__init__()
        self.stride = stride
        kernel_size = kernel_size
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride == 2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)
        return X_h, X_l


class LastOctave(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, pad_type='zero', norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(LastOctave, self).__init__()
        self.stride = stride
        kernel_size = kernel_size
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_l2h = self.l2h(X_l)
        X_h2h = self.h2h(X_h)
        X_l2h = self.upsample(X_l2h)

        X_h = X_h2h + X_l2h
        return X_h


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero',
                 norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc + gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc + 2 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc + 3 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc + 4 * gc, nc, 3, stride, bias=bias, pad_type=pad_type,
                                norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero',
                 norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type,
                                          norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type,
                                          norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type,
                                          norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


class octave_ResidualDenseBlockTiny_4C(nn.Module):
    '''
    Residual Dense Block
    style: 4 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, alpha=0.5, stride=1, bias=True, pad_type='zero',
                 norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(octave_ResidualDenseBlockTiny_4C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = Octave(nc, gc, kernel_size, alpha, stride, bias=bias, pad_type=pad_type,
                            norm_type=norm_type, act_type=act_type, mode=mode)
        # conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
        #     norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = Octave(nc + gc, gc, kernel_size, alpha, stride, bias=bias, pad_type=pad_type,
                            norm_type=norm_type, act_type=act_type, mode=mode)
        # conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
        #     norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = Octave(nc + 2 * gc, gc, kernel_size, alpha, stride, bias=bias, pad_type=pad_type,
                            norm_type=norm_type, act_type=act_type, mode=mode)
        # conv_block(nc+level 2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
        #     norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv4 = Octave(nc + 3 * gc, nc, kernel_size, alpha, stride, bias=bias, pad_type=pad_type,
                            norm_type=norm_type, act_type=act_type, mode=mode)
        # conv_block(nc+3*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
        #     norm_type=norm_type, act_type=last_act, mode=mode)
        self.average = nn.AdaptiveAvgPool2d(1)
        self.attention = sequential(
            conv_block(nc, nc // 16, 1, stride, 1, 1, bias, pad_type,
                       norm_type, act_type, mode),
            conv_block(nc // 16, nc, 1, stride, 1, 1, bias, pad_type,
                       norm_type, None, mode),
            nn.Sigmoid())

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2((torch.cat((x[0], x1[0]), dim=1), (torch.cat((x[1], x1[1]), dim=1))))
        x3 = self.conv3((torch.cat((x[0], x1[0], x2[0]), dim=1), (torch.cat((x[1], x1[1], x2[1]), dim=1))))
        x4 = self.conv4(
            (torch.cat((x[0], x1[0], x2[0], x3[0]), dim=1), (torch.cat((x[1], x1[1], x2[1], x3[1]), dim=1))))

        res = (x4[0].mul(0.2), x4[1].mul(0.2))
        x = (x[0] + res[0], x[1] + res[1])
        X_hpooling = self.average(x[0])
        X_lpooling = self.average(x[1])
        X_attention = torch.cat((X_hpooling, X_lpooling), dim=1)
        X_attention = self.attention(X_attention)
        X_h_attention = X_attention[:, :32, :, :]
        X_l_attention = X_attention[:, 32:, :, :]
        X_h = x[0] * X_h_attention
        X_l = x[1] * X_l_attention
        x = (X_h, X_l)
        return x


class octave_RRDBTiny(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, alpha=0.5, bias=True, pad_type='zero',
                 norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(octave_RRDBTiny, self).__init__()
        self.RDB1 = octave_ResidualDenseBlockTiny_4C(nc=nc, kernel_size=kernel_size, alpha=alpha, gc=gc, stride=stride,
                                                     bias=bias, pad_type=pad_type,
                                                     norm_type=norm_type, act_type=act_type, mode=mode)
        self.RDB2 = octave_ResidualDenseBlockTiny_4C(nc=nc, kernel_size=kernel_size, alpha=alpha, gc=gc, stride=stride,
                                                     bias=bias, pad_type=pad_type,
                                                     norm_type=norm_type, act_type=act_type, mode=mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)

        res = (out[0].mul(0.2), out[1].mul(0.2))
        x = (x[0] + res[0], x[1] + res[1])
        return x


class channel_attention_OctaveConv(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size, alpha=0.5, stride=1, dilation=1, groups=1,
                 bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA', reduction=8):
        super(channel_attention_OctaveConv, self).__init__()
        assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
        padding = get_valid_padding(kernel_size, dilation) if pad_type == 'zero' else 0

        self.stride = stride
        self.out = out_nc
        self.alpha = alpha

        self.l2l = nn.Conv2d(in_nc // 2, out_nc, kernel_size, 1, padding, dilation, groups, bias)

        self.h2h = nn.Conv2d(in_nc // 2, out_nc, kernel_size, 1, padding, dilation, groups, bias)
        self.a = act(act_type) if act_type else None
        self.n_h = norm(norm_type, out_nc) if norm_type else None
        self.n_l = norm(norm_type, out_nc) if norm_type else None

        self.average = nn.AdaptiveAvgPool2d(1)

        self.attention = sequential(
            conv_block(in_nc, in_nc // reduction, 1, stride, dilation, groups, bias, pad_type,
                       norm_type, act_type, mode),
            conv_block(in_nc // reduction, out_nc, 1, stride, dilation, groups, bias, pad_type,
                       norm_type, None, mode),
            nn.Sigmoid())

    def forward(self, x):
        X_h, X_l = x

        X_h2h = self.h2h(X_h)
        X_l2l = self.l2l(X_l)
        X_hpooling = self.average(X_h2h)
        X_lpooling = self.average(X_l2l)
        X_attention = torch.cat((X_hpooling, X_lpooling), dim=1)
        X_attention = self.attention(X_attention)
        X_h_attention = X_attention[:, :self.out - int(self.alpha * self.out), :, :]
        X_l_attention = X_attention[:, self.out - int(self.alpha * self.out):, :, :]

        # print(X_l2h.shape,"~~~~",X_h2h.shape)
        X_h = X_h2h * X_h_attention
        X_l = X_l2l * X_l_attention

        if self.n_h and self.n_l:
            X_h = self.n_h(X_h)
            X_l = self.n_l(X_l)

        if self.a:
            X_h = self.a(X_h)
            X_l = self.a(X_l)

        return X_h, X_l


####################
# Upsampler
####################


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                       pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias,
                      pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                 pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias,
                      pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)
