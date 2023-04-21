import torch.nn as nn
import numpy as np

'''
自定义的卷积操作
使用了边界反射填充 ReflectionPad2d
'''
class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        reflection_padding = int(np.floor(kernel_size/2))
        self.reflection_pad = nn.ReflectionPad2d(padding = reflection_padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        return out