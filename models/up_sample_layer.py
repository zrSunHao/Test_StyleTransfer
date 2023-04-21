import torch as t
import torch.nn as nn

# 上采样 对应 VGG19 中的 pool/2
class UpSampleLayer(nn.Module):

    def __init__(self, in_channels):
        super(UpSampleLayer, self).__init__()
        self.input = nn.Parameter(t.randn(in_channels))

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2)