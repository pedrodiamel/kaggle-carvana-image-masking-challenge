import torch.nn as nn

from ptsemseg.models.utils import *

class unet(nn.Module):

    def __init__(self, feature_scale=1, n_classes=21, is_deconv=False, in_channels=3, is_batchnorm=True):
        super(unet, self).__init__()
        #self.is_deconv = is_deconv
        self.is_deconv = False
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        #filters = [x / self.feature_scale for x in filters]

        self.down1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.up4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        down1,befdown1 = self.down1(inputs)
        down2,befdown2 = self.down2(down1)
        down3,befdown3 = self.down3(down2)
        down4,befdown4 = self.down4(down3)
        center = self.center(down4)
        up4 = self.up4(befdown4, center)
        up3 = self.up3(befdown3, up4)
        up2 = self.up2(befdown2, up3)
        up1 = self.up1(befdown1, up2)
        
        return self.final(up1)
