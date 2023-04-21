import torch.nn as nn

from .conv_layer import ConvLayer
from .up_sample_layer import UpSampleLayer

class Deocder(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.convL1 = ConvLayer(in_channels=512, out_channels=256, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.up1 = UpSampleLayer(256)

        self.convL2_1 = ConvLayer(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.relu2_1 = nn.ReLU()
        self.convL2_2 = ConvLayer(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.relu2_2 = nn.ReLU()
        self.convL2_3 = ConvLayer(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.relu2_3 = nn.ReLU()
        self.convL2_4 = ConvLayer(in_channels=256, out_channels=128, kernel_size=3, stride=1)
        self.relu2_4 = nn.ReLU()
        self.up2 = UpSampleLayer(128)

        self.convL3_1 = ConvLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.relu3_1 = nn.ReLU()
        self.convL3_2 = ConvLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1)
        self.relu3_2 = nn.ReLU()
        self.up3 = UpSampleLayer(128)

        self.convL4_1 = ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu4_1 = nn.ReLU()
        self.convL4_2 = ConvLayer(in_channels=64, out_channels=3, kernel_size=3, stride=1)

    
    def forward(self, x):
        out = self.convL1(x); out = self.relu1(out)
        out = self.up1(out)

        out = self.convL2_1(out); out = self.relu2_1(out)
        out = self.convL2_2(out); out = self.relu2_2(out)
        out = self.convL2_3(out); out = self.relu2_3(out)
        out = self.convL2_4(out); out = self.relu2_4(out)
        out = self.up2(out)

        out = self.convL3_1(out); out = self.relu3_1(out)
        out = self.convL3_2(out); out = self.relu3_2(out)
        out = self.up3(out)

        out = self.convL4_1(out); out = self.relu4_1(out)
        out = self.convL4_2(out)

        return out
