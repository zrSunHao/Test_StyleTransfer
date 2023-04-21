import torch as t
import torch.nn as nn
import numpy as np

from .vgg_19 import VGG19
from .decoder import Deocder
from tools import AdaIn, get_conetnt_loss, get_style_loss

class TransferNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VGG19()
        self.decoder = Deocder()
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def generate(self, content, style):
        content_feature = self.encoder(content).relu4_1
        style_feature = self.encoder(style).relu4_1
        adain = AdaIn(content_feature, style_feature)
        return self.decoder(adain)
    
    def forward(self, content, style):
        content_feature = self.encoder(content).relu4_1
        style_feature = self.encoder(style).relu4_1
        adain = AdaIn(content_feature, style_feature)
        output = self.decoder(adain)

        output_features = self.encoder(output).relu4_1
        content_mid = self.encoder(output)
        style_mid = self.encoder(style)

        content_loss = get_conetnt_loss(output_features, adain)
        style_loss = get_style_loss(content_mid, style_mid)

        return content_loss + 10 * style_loss        # 将损失进行加权



