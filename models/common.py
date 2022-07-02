import math
import torch
import torchvision
import torch.nn as nn
import torch.fft as fft
import PIL.Image as Image


def fft2d(input):
    fft_out = fft.fftn(input, dim=(2, 3), norm='ortho')
    return fft_out


def fftshift2d(input):
    b, c, h, w = input.shape
    fs11 = input[:, :, -h // 2:h, -w // 2:w]
    fs12 = input[:, :, -h // 2:h, 0:w // 2]
    fs21 = input[:, :, 0:h // 2, -w // 2:w]
    fs22 = input[:, :, 0:h // 2, 0:w // 2]
    output = torch.cat([torch.cat([fs11, fs21], axis=2), torch.cat([fs12, fs22], axis=2)], axis=3)
    output = torchvision.transforms.Resize((128, 128), interpolation=Image.BICUBIC)(output)
    return output


def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


def conv3(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
        m.append(nn.ReLU(True))
        m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ConvBlock(nn.Module):
    def __init__(self, conv, in_channel, out_channel, kernel_size, bias=True):
        super().__init__()
        m = []
        m.append(conv(in_channel, out_channel, kernel_size, bias=bias))
        m.append(nn.LeakyReLU(negative_slope=0.1))
        m.append(conv(out_channel, out_channel, kernel_size, bias=bias))
        m.append(nn.LeakyReLU(negative_slope=0.1))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(act)
                m.append(nn.PixelShuffle(2))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)
