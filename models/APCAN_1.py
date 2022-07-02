import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import conv, fft2d, fftshift2d, Upsampler


class CALayer(nn.Module):
    def __init__(self, n_feat, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat // reduction, n_feat, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ACALayer(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act=nn.ReLU()):
        super().__init__()
        self.amplitude_conv = conv(n_feat, n_feat, kernel_size)
        self.act = act
        self.global_average_pooling2d = nn.AdaptiveAvgPool2d(1)
        self.global_max_pooling2d = nn.AdaptiveMaxPool2d(1)
        self.amplitude_attention = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_fft = fft2d(x)
        x_amplitude = torch.abs(x_fft)
        x_amplitude = torch.pow(x_amplitude + 1e-8, 0.8)
        amplitude1 = fftshift2d(x_amplitude)
        amplitude2 = self.act(self.amplitude_conv(amplitude1))
        y_avg = self.global_average_pooling2d(amplitude2).view(b, c, 1, 1)
        y_max = self.global_max_pooling2d(amplitude2).view(b, c, 1, 1)
        y = torch.cat([y_avg, y_max], dim=1)
        y = self.amplitude_attention(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        output = x * y
        return output


class PCALayer(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act=nn.ReLU()):
        super().__init__()
        self.phase_conv = conv(n_feat, n_feat, kernel_size)
        self.act = act
        self.global_average_pooling2d = nn.AdaptiveAvgPool2d(1)
        self.global_max_pooling2d = nn.AdaptiveMaxPool2d(1)
        self.phase_attention = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_fft = fft2d(x)
        x_phase = torch.atan2(x_fft.imag + 1e-8, x_fft.real + 1e-8)
        phase1 = fftshift2d(x_phase)
        phase2 = self.act(self.phase_conv(phase1))
        y_avg = self.global_average_pooling2d(phase2).view(b, c, 1, 1)
        y_max = self.global_max_pooling2d(phase2).view(b, c, 1, 1)
        y = torch.cat([y_avg, y_max], dim=1)
        y = self.phase_attention(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        output = x * y
        return output


class APCALayer(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act=nn.ReLU()):
        super().__init__()
        self.K = 2
        self.T = n_feat
        self.act = act
        self.conv = conv(n_feat, n_feat, 1)
        self.amplitude_attention = ACALayer(conv, n_feat, kernel_size, reduction)
        self.phase_attention = PCALayer(conv, n_feat, kernel_size, reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Sequential(
            conv(n_feat, n_feat // reduction, kernel_size=1),
            nn.ReLU(),
            conv(n_feat // reduction, self.K, kernel_size=1))

    def forward(self, x):
        b, c, h, w = x.shape
        amplitude_map = self.amplitude_attention(x)
        phase_map = self.phase_attention(x)
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.weight(y)
        ax = F.softmax(y / self.T, dim=1)
        alpha, _ = torch.min(ax, dim=1)
        beta, _ = torch.max(ax, dim=1)
        output = amplitude_map * alpha.view(b, 1, 1, 1) + phase_map * beta.view(b, 1, 1, 1)
        output = self.act(self.conv(output))
        output += x
        return output


class APCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU()):
        super(APCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(APCALayer(conv, n_feat, kernel_size, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            APCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act) for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class APCAN(nn.Module):
    def __init__(self, opt):
        super(APCAN, self).__init__()
        n_resgroups = 4
        n_resblocks = 4
        n_feats = 64
        kernel_size = 3
        reduction = 16
        act = nn.ReLU()
        modules_head = [conv(9, n_feats, kernel_size)]
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_tail = [Upsampler(conv, opt.scale, n_feats, act=act),
                        conv(n_feats, opt.nch_out, kernel_size)]
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x[:, 0:9, :, :])
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x
