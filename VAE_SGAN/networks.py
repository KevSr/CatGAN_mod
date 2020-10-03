from __future__ import print_function
import os
import random
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models

import numpy as np
import random

from functools import partial
from torch.autograd import grad as torch_grad
from PIL import Image

import math
from math import floor, log2
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset

class ClippedReLU(nn.Module):
    def __init__(self):
        super(ClippedReLU, self).__init__()
        self.clipped = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.clipped(x)
        return x.clamp(max=10)

class inception(nn.Module):
    def __init__(self, in_channel, acti, filter1, filter3r, filter3, filter5r, filter5, filterpool):
        super(inception, self).__init__()
        if acti == 'relu':
          self.acti = nn.ReLU(inplace=True)
        elif acti == 'leaky':
          self.acti = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        elif acti == 'clipped':
          self.acti = ClippedReLU()

        self.conv1 = nn.Conv2d(in_channel, filter1, 1, 1)
        self.norm1 = nn.BatchNorm2d(filter1)
        self.conv3r = nn.Conv2d(in_channel, filter3r, 1, 1)
        self.norm3r = nn.BatchNorm2d(filter3r)
        self.conv3 = nn.Conv2d(filter3r, filter3, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(filter3)
        self.conv5r = nn.Conv2d(in_channel, filter5r, 1, 1)
        self.norm5r = nn.BatchNorm2d(filter5r)
        self.conv5 = nn.Conv2d(filter5r, filter5, 5, 1, 2)
        self.norm5 = nn.BatchNorm2d(filter5)
        self.pool = nn.MaxPool2d(3, 1, 1)
        self.convpool = nn.Conv2d(in_channel, filterpool, 1, 1)
        self.normpool = nn.BatchNorm2d(filterpool)

    def forward(self, x):
        x1 = self.acti(self.norm1(self.conv1(x)))
        x2 = self.acti(self.norm3r(self.conv3r(x)))
        x2 = self.acti(self.norm3(self.conv3(x2)))
        x3 = self.acti(self.norm5r(self.conv5r(x)))
        x3 = self.acti(self.norm5(self.conv5(x3)))
        x4 = self.acti(self.normpool(self.convpool(self.pool(x))))
        x = torch.cat((x1, x2, x3, x4), 1)
        return x


class identity_block(nn.Module):
    def __init__(self, in_channel, acti, filter1, filter2, filter3):
        super(identity_block, self).__init__()
        if acti == 'relu':
          self.acti = nn.ReLU(inplace=True)
        elif acti == 'leaky':
          self.acti = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        elif acti == 'clipped':
          self.acti = ClippedReLU()

        self.conv1 = nn.Conv2d(in_channel, filter1, 1, 1)
        self.norm1 = nn.BatchNorm2d(filter1)
        self.conv2 = nn.Conv2d(filter1, filter2, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(filter2)
        self.conv3 = nn.Conv2d(filter2, filter3, 1, 1)
        self.norm3 = nn.BatchNorm2d(filter3)

    def forward(self, x):
        x1 = self.acti(self.norm1(self.conv1(x)))
        x1 = self.acti(self.norm2(self.conv2(x1)))
        x1 = self.acti(self.norm3(self.conv3(x1)))
        x = self.acti(torch.add(x1,x))
        return x


class conv_block(nn.Module):
    def __init__(self, in_channel, acti, filter1, filter2, filter3, stride = (2, 2)):
        super(conv_block, self).__init__()
        if acti == 'relu':
          self.acti = nn.ReLU(inplace=True)
        elif acti == 'leaky':
          self.acti = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        elif acti == 'clipped':
          self.acti = ClippedReLU()
        
        self.convshort = nn.Conv2d(in_channel, filter3, 1, stride)
        self.normshort = nn.BatchNorm2d(filter3)

        self.conv1 = nn.Conv2d(in_channel, filter1, 1, stride)
        self.norm1 = nn.BatchNorm2d(filter1)
        self.conv2 = nn.Conv2d(filter1, filter2, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(filter2)
        self.conv3 = nn.Conv2d(filter2, filter3, 1, 1)
        self.norm3 = nn.BatchNorm2d(filter3)

    def forward(self, x):
        x1 = self.normshort(self.convshort(x))
        x2 = self.acti(self.norm1(self.conv1(x)))
        x2 = self.acti(self.norm2(self.conv2(x2)))
        x2 = self.acti(self.norm3(self.conv3(x2)))
        x = self.acti(torch.add(x1, x2))
        return x

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class Mapping(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1, acti = 'leaky'):
        super().__init__()
        if acti == 'relu':
          acti = nn.ReLU(inplace=True)
        elif acti == 'leaky':
          acti = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        elif acti == 'clipped':
         acti = ClippedReLU()
       
        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), acti])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        
        return self.net(x)


class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if prev_rgb is not None:
            x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)

        return x

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, basic =False, acti= 'leaky', **kwargs):
        super().__init__()

        if acti == 'relu':
          acti = nn.ReLU(inplace=True)
        elif acti == 'leaky':
          acti = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        elif acti == 'clipped':
         acti = ClippedReLU()


        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.basic = basic
        self.chan = in_chan

        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in')


    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y = None, device = None):
        b, c, h, w = x.shape

        if self.basic:
            padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
            weights = nn.Parameter(torch.randn((b, self.filters, self.chan, self.kernel, self.kernel)).to(device=device))
            nn.init.kaiming_normal_(weights, a=0, mode='fan_in')
            _, _, *ws = weights.shape
            weights = weights.reshape(b*self.filters, *ws)
            x = x.reshape(1, -1, h, w)
            x = F.conv2d(x, weights, padding=padding, groups=b)
            x = x.reshape(-1, self.filters, h, w) 
            return x

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + 1e-6)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)

        return x

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, architecture, acti= 'leaky', upsample = True, upsample_rgb = True, rgba = False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None
        self.architecture = architecture
        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)
        self.resconv = Conv2DMod(input_channels, filters, 1, basic = True)

        if acti == 'relu':
          self.activation = nn.ReLU(inplace=True)
        elif acti == 'leaky':
          self.activation = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        elif acti == 'clipped':
          self.activation = ClippedReLU()

        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)
    def forward(self, x, prev_rgb, istyle, inoise, device):
        t = x
        
        if self.upsample is not None:
            x = self.upsample(x)
            
        inoise = inoise[:, :x.shape[3], :x.shape[2], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))
        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)
        
        if self.architecture == 'resnet':
            if self.upsample is not None:
                t = self.upsample(t)
            t = self.resconv(t, device= device)
            x = (x + t) * (1 / np.sqrt(2))
        
        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb

class StyleGAN(nn.Module):
    def __init__(self, image_size, nz, device, style_depth = 8, latent_dim = 512, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512, acti = 'leaky'):
        super(StyleGAN, self).__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(16) - 1)
        self.device = device
        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(nz, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 15, 12)))
            

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])

        self.mapp = Mapping(latent_dim, style_depth, lr_mul = 0.1, acti = acti)

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                'resnet',
                acti,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent                
            )
            self.blocks.append(block)

    def latent_to_w(self, Mapping, latent_descr):
        return [(Mapping(z), num_layers) for z, num_layers in latent_descr]

    def styles_def_to_tensor(self, styles_def):
        return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)


    def forward(self, styles, input_noise):
        stylez = self.latent_to_w(self.mapp, styles)

        style = self.styles_def_to_tensor(stylez)

        batch_size = style.shape[0]
        image_size = self.image_size


        if self.no_const:
            avg_style = style.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        style_ = style.transpose(0, 1)
        x = self.initial_conv(x)
        for style_, block in zip(style_, self.blocks):

            x, rgb = block(x, rgb, style_, input_noise, self.device)

        return style, rgb

class VAE(nn.Module):
    def __init__(self, nz, device, acti = 'leaky'):
        super(VAE, self).__init__()
        if acti == 'relu':
          self.acti = nn.ReLU(inplace=True)
        elif acti == 'leaky':
          self.acti = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        elif acti == 'clipped':
          self.acti = ClippedReLU()

        self.nz = nz
        self.device = device
        # input is Z, going into a convolution
        self.conv1 = nn.Conv2d(3, 96, 5, 1, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(96)
        # self.pool1 = nn.MaxPool2d(3, 2, 1)

        self.conv2 = nn.Conv2d(96, 96, 5, 1, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(96)
        self.pool2 = nn.MaxPool2d(3, 2, 1)

        self.conv3 = nn.Conv2d(96, 192, 5, 1, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(192)
        # self.pool3 = nn.MaxPool2d(3, 2, 1)

        self.conv4 = nn.Conv2d(192, 192, 5, 1, 2, bias=False)
        self.bn4 = nn.BatchNorm2d(192)
        self.pool4 = nn.MaxPool2d(3, 2, 1)

        self.mu = nn.Linear(15*12*192 , nz)
        self.sigma = nn.Linear(15*12*192 , nz)

        self.decode = Generator(nz, acti)

    def reparamatize(self, mu, logvar, batch):
        std = torch.exp(0.5*logvar)
        eps = torch.randn(batch, self.nz, device=self.device)
        return eps.mul(std).add_(mu)

    def forward(self, x, batch):
        x = self.bn1(self.conv1(x))

        # x = self.pool1(self.acti(x))
        x = self.acti(x)
        x = self.bn2(self.conv2(x))
        x = self.pool2(self.acti(x))
        x = self.bn3(self.conv3(x))
        # x = self.pool3(self.acti(x))
        x = self.acti(x)
        x = self.bn4(self.conv4(x))
        x = self.pool4(self.acti(x))
        x = x.view(-1, 15*12*192)
        mu = self.mu(x)
        logvar = self.sigma(x)
        out = self.reparamatize(mu, logvar, batch)
        out = self.decode(out)
        return mu, logvar, out


class Generator(nn.Module):
    def __init__(self, nz, acti = 'leaky'):
        super(Generator, self).__init__()
        if acti == 'relu':
          self.acti = nn.ReLU(inplace=True)
        elif acti == 'leaky':
          self.acti = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        elif acti == 'clipped':
          self.acti = ClippedReLU()

        # input is Z, going into a convolution
        self.lin = nn.Linear(nz, 192*15*12)
        self.bn = nn.BatchNorm1d(192*15*12)
        self.up = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(192, 192, 5, 1, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(192)
        self.conv2 = nn.Conv2d(192, 96, 5, 1, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 96, 5, 1, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 3, 5, 1, 2, bias=False)

        self.tanh = nn.Sigmoid()#nn.Tanh()
        # self.lin = nn.ConvTranspose2d( nz, 1024, (4, 3), 1, 0, bias=False)
        # self.bn = nn.BatchNorm2d(1024)

        # self.conv1 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        # self.bn1 = nn.BatchNorm2d(512)

        # self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        # self.bn2 = nn.BatchNorm2d(256)

        # self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        # self.bn3 = nn.BatchNorm2d(128)

        # self.conv4 = nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False)
        # self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.acti(self.bn(self.lin(x)))
        x = self.up(x.view(-1, 192, 15, 12))
        x = self.acti(self.bn1(self.conv1(x)))
        x = self.acti(self.bn2(self.conv2(x)))
        x = self.up(x)
        x = self.acti(self.bn3(self.conv3(x)))
        # x = self.conv4(x)
        x = self.tanh(self.conv4(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, K, acti = 'leaky'):
        super(Discriminator, self).__init__()
        if acti == 'relu':
          self.acti = nn.ReLU(inplace=True)
        elif acti == 'leaky':
          self.acti = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        elif acti == 'clipped':
          self.acti = ClippedReLU()

        self.conv1 = nn.Conv2d(3, 96, 5, 1, 2)
        self.norm1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 96, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 96, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(96, 192, 3, 1, 1)
        self.norm4 = nn.BatchNorm2d(192)
        self.conv5 = nn.Conv2d(192, 192, 3, 1, 1)
        self.norm5 = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(192, 192, 3, 1, 1)
        self.norm6 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(3, 2, 1)
        self.conv7 = nn.Conv2d(192, 192, 3, 1, 1)
        self.norm7 = nn.BatchNorm2d(192)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.norm8 = nn.BatchNorm2d(192)
        self.conv9 = nn.Conv2d(192, 10, 1)
        self.norm9 = nn.BatchNorm2d(10)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(10*15*12, K)

    def forward(self, x):
        x = self.acti(self.norm1(self.conv1(x)))
        x = self.acti(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        x = self.pool1(self.acti(x))
        x = self.acti(self.norm4(self.conv4(x)))
        x = self.acti(self.norm5(self.conv5(x)))
        x = self.norm6(self.conv6(x))
        x = self.pool2(self.acti(x))
        x = self.acti(self.norm7(self.conv7(x)))
        x = self.acti(self.norm8(self.conv8(x)))
        x = self.acti(self.norm9(self.conv9(x)))
        x = x.view(-1, 10*15*12)
        x = self.softmax(self.fc(x))

        return x




class Alexnet(nn.Module):

    """
    # Reference:
    - [ImageNet classification with deep convolutional neural networks]
        (https://doi.org/10.1145/3065386)
    """
  
    def __init__(self, K, acti = 'leaky'):
        super(Alexnet, self).__init__()

        if acti == 'relu':
          self.acti = nn.ReLU(inplace=True)
        elif acti == 'leaky':
          self.acti = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        elif acti == 'clipped':
          self.acti = ClippedReLU()

        self.conv1 = nn.Conv2d(3, 96, 11, 4)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.norm1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(3, 2, 1)
        self.norm2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.norm4 = nn.BatchNorm2d(384)
        self.conv5 = nn.Conv2d(384, 384, 3, 1, 1)
        self.norm5 = nn.BatchNorm2d(384)
        self.pool3 = nn.MaxPool2d(3, 2, 1)

        self.fc1 = nn.Linear(384*2*2,4096)
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(4096, K)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.norm1(self.conv1(x))
        x = self.pool1(self.acti(x))
        x = self.norm2(self.conv2(x))
        x = self.pool2(self.acti(x))
        x = self.acti(self.norm3(self.conv3(x)))
        x = self.acti(self.norm4(self.conv4(x)))
        x = self.norm5(self.conv5(x))
        x = self.pool3(self.acti(x))
        x = x.view(-1,384*2*2)
        x = self.drop1(self.acti(self.fc1(x)))
        x = self.drop2(self.acti(self.fc2(x)))
        x = self.softmax((self.fc3(x)))
        return x


class VGG_16(nn.Module):

    """
    # Reference:
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition]
        (https://arxiv.org/pdf/1409.1556)
    """

    def __init__(self, K, acti = 'leaky'):
        super(VGG_16, self).__init__()

        if acti == 'relu':
          self.acti = nn.ReLU(inplace=True)
        elif acti == 'leaky':
          self.acti = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        elif acti == 'clipped':
          self.acti = ClippedReLU()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.norm4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.norm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.norm6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.norm7 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv8 = nn.Conv2d(256, 512, 3, 1, 1)
        self.norm8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, 3, 1, 1)
        self.norm9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, 1)
        self.norm10 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)


        self.conv11 = nn.Conv2d(512, 512, 3, 1, 1)
        self.norm11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 1)
        self.norm12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 1)
        self.norm13 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512*1*1,4096)
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(4096, K)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        x = self.acti(self.norm1(self.conv1(x)))
        x = self.acti(self.norm2(self.conv2(x)))
        x = self.pool1(x)
        x = self.acti(self.norm3(self.conv3(x)))
        x = self.acti(self.norm4(self.conv4(x)))
        x = self.pool2(x)
        x = self.acti(self.norm5(self.conv5(x)))
        x = self.acti(self.norm6(self.conv6(x)))
        x = self.acti(self.norm7(self.conv7(x)))
        x = self.pool3(x)
        x = self.acti(self.norm8(self.conv8(x)))
        x = self.acti(self.norm9(self.conv9(x)))
        x = self.acti(self.norm10(self.conv10(x)))
        x = self.pool4(x)
        x = self.acti(self.norm11(self.conv11(x)))
        x = self.acti(self.norm12(self.conv12(x)))
        x = self.acti(self.norm13(self.conv13(x)))
        x = self.pool5(x)
        x = x.view(-1,512*1*1)
        x = self.drop1(self.acti(self.fc1(x)))
        x = self.drop2(self.acti(self.fc2(x)))
        x = self.softmax(self.fc3(x))      
        return x


class ResNet_50(nn.Module):

    """
    # Reference:
    - [Deep Residual Learning for Image Recognition]
        (https://arxiv.org/abs/1512.03385)
    """

    def __init__(self, K, acti = 'leaky'):
        super(ResNet_50, self).__init__()

        if acti == 'relu':
          self.acti = nn.ReLU(inplace=True)
        elif acti == 'leaky':
          self.acti = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        elif acti == 'clipped':
          self.acti = ClippedReLU()
          
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.norm1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2, 1)

        self.conv_a1 = conv_block(64, acti, 64, 64, 256, (1,1))
        self.iden_a1 = identity_block(256, acti, 64, 64, 256)
        self.iden_a2 = identity_block(256, acti, 64, 64, 256)

        self.conv_b1 = conv_block(256, acti, 128, 128, 512)
        self.iden_b1 = identity_block(512, acti, 128, 128, 512)
        self.iden_b2 = identity_block(512, acti, 128, 128, 512)
        self.iden_b3 = identity_block(512, acti, 128, 128, 512)

        self.conv_c1 = conv_block(512, acti, 256, 256, 1024)
        self.iden_c1 = identity_block(1024, acti, 256, 256, 1024)
        self.iden_c2 = identity_block(1024, acti, 256, 256, 1024)
        self.iden_c3 = identity_block(1024, acti, 256, 256, 1024)
        self.iden_c4 = identity_block(1024, acti, 256, 256, 1024)
        self.iden_c5 = identity_block(1024, acti, 256, 256, 1024)

        self.conv_d1 = conv_block(1024, acti, 512, 512, 2048)
        self.iden_d1 = identity_block(2048, acti, 512, 512, 2048)
        self.iden_d2 = identity_block(2048, acti, 512, 512, 2048)

        #Pooling kernel reduced due to the dimension of our dataset - [*, *, 2, 2]
        self.avgpool = nn.AvgPool2d(2, 1)
        self.fc = nn.Linear(2048,K)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
      x = self.norm1(self.conv1(x))
      x = self.pool(self.acti(x))
      x = self.iden_a2(self.iden_a1(self.conv_a1(x)))
      x = self.iden_b3(self.iden_b2(self.iden_b1(self.conv_b1(x))))
      x = self.iden_c2(self.iden_c1(self.conv_c1(x)))
      x = self.iden_c5(self.iden_c4(self.iden_c3(x)))
      x = self.iden_d2(self.iden_d1(self.conv_d1(x)))
      x = self.avgpool(x)
      x = x.view(-1, 2048)
      x = self.softmax(self.fc(x))
      
      return x


class GoogleNet(nn.Module):

    """
    # Reference:
    - [Going Deeper with Convolutions]
        (https://arxiv.org/pdf/1409.4842)
    """

    def __init__(self, K, acti = 'leaky'):
        super(GoogleNet, self).__init__()

        if acti == 'relu':
          self.acti = nn.ReLU(inplace=True)
        elif acti == 'leaky':
          self.acti = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        elif acti == 'clipped':
          self.acti = ClippedReLU()

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 1, 1)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 192, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(3, 2, 1)

        self.incep3a = inception(192, acti, 64, 96, 128, 16, 32, 32)
        self.incep3b = inception(256, acti, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(3, 2, 1)

        self.incep4a = inception(480, acti, 192, 96, 208, 16, 48 ,64)
        self.incep4b = inception(512, acti, 160, 112, 224, 24, 64, 64)
        self.incep4c = inception(512, acti, 128, 128, 256, 24, 64, 64)
        self.incep4d = inception(512, acti, 112, 144, 288, 32, 64, 64)
        self.incep4e = inception(528, acti, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(3, 2, 1)

        self.incep5a = inception(832, acti, 256, 160, 320, 32, 128, 128)
        self.incep5b = inception(832, acti, 384, 192, 384, 48, 128, 128)

        #Pooling kernel reduced due to the dimension of our dataset - [*, *, 4, 3]
        self.avgpool4a = nn.AvgPool2d(3, 3)
        self.conv4a = nn.Conv2d(512, 128, 1, 1)
        self.fc4a = nn.Linear(128, 1024)
        self.drop4a = nn.Dropout(0.7)
        self.fc4a_1 = nn.Linear(1024, K)

        #Pooling kernel reduced due to the dimension of our dataset - [*, *, 4, 3]
        self.avgpool4d = nn.AvgPool2d(3, 3)
        self.conv4d = nn.Conv2d(528, 128, 1, 1)
        self.fc4d = nn.Linear(128, 1024)
        self.drop4d = nn.Dropout(0.7)
        self.fc4d_1 = nn.Linear(1024, K)

        #Pooling kernel reduced due to the dimension of our dataset - [*, *, 2, 2]
        self.avgpool5b = nn.AvgPool2d(2, 1)
        self.drop5b = nn.Dropout(0.4)
        self.fc5b = nn.Linear(1024,K)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.norm1(self.conv1(x))
        x = self.pool1(self.acti(x))
        x = self.acti(self.norm2(self.conv2(x)))
        x = self.acti(self.norm3(self.conv3(x)))
        x = self.pool2(x)

        x3 = self.incep3b(self.incep3a(x))
        x3 = self.pool3(x3)

        x4a = self.incep4a(x3)

        x4 = self.incep4c(self.incep4b(x4a))
        x4p = self.incep4d(x4)

        x4 = self.pool4(self.incep4e(x4p))
        x5 = self.incep5b(self.incep5a(x4))

        out1 = self.avgpool4a(x4a)
        out1 = self.acti(self.conv4a(out1))
        out1 = out1.view(-1, 128)
        out1 = self.acti(self.fc4a(out1))
        out1 = self.fc4a_1(self.drop4a(out1))
        out1 = self.softmax(out1)

        out2 = self.avgpool4d(x4p)
        out2 = self.acti(self.conv4d(out2))
        out2 = out2.view(-1, 128)
        out2 = self.acti(self.fc4d(out2))
        out2 = self.fc4d_1(self.drop4d(out2))
        out2 = self.softmax(out2)

        out3 = self.avgpool5b(x5)
        out3 = out3.view(-1, 1024)
        out3 = self.fc5b(self.drop5b(out3))
        out3 = self.softmax(out3)

        return out3


def disc_sel(option, K, device, acti = 'leaky'):
    if option == 'catgan':
        return Discriminator(K, acti).to(device)#.half()
    elif option == 'alex':
        return Alexnet(K, acti).to(device)
    elif option == 'vgg':
        return VGG_16(K, acti).to(device)
    elif option == 'resnet':
        return ResNet_50(K, acti).to(device)
    elif option == 'google':
        return GoogleNet(K, acti).to(device)
    else: 
      print('No option')


def gen_sel(option, nz, device, acti = 'leaky'):
    if option == 'catgan':
        return Generator(nz, acti).to(device)
    elif option == 'vae':
        return VAE(nz, device, acti).to(device)
    elif option == 'style':
        return StyleGAN(image_size = 60, nz = nz, device = device, style_depth = 8, latent_dim = 512, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512, acti = 'leaky').to(device)
    else: 
      print('No option')


def opti_sel(model, option, lr):
  if option == 'adam':
    return optim.Adam(model.parameters(), lr = lr, eps = 1e-7)
  elif option == 'rmsprop':
    return optim.RMSprop(model.parameters(), lr = lr, eps = 1e-7, alpha = 0.9)
  elif option == 'sgdm':
    return optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
