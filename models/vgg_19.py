import torch as t
import torch.nn as nn
from torchvision.models import vgg19,VGG19_Weights
from collections import namedtuple

class VGG19(nn.Module):

    def __init__(self):
        super().__init__()
        features = list(vgg19(weights=VGG19_Weights.DEFAULT).features)[:21]
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, layer in enumerate(self.features):
            x = layer(x)
            if ii in {1, 6, 11, 20}:
                results.append(x)
        outputs = namedtuple('VggOutputs', ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
        return outputs(*results)