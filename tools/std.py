import torch as t
import torchvision as tv
import numpy as np

'''
计算：均值、标准差
'''
def calc_mean_std(features):
    batch_size, c = features.size()[:2]

    # features_mean: 512*1*1
    features_mean = features.reshape(batch_size, c, -1).mean(dim = 2)
    features_mean = features_mean.reshape(batch_size, c, 1, 1)

    # features_std: 512*1*1
    features_std = features.reshape(batch_size, c, -1).std(dim = 2)
    features_std = features_std.reshape(batch_size, c, 1, 1)
    features_std = features_std + 1e-16

    return features_mean, features_std


def AdaIn(x, y):
    # x: 512*53*80, y: 512*99*125
    x_mean, x_std = calc_mean_std(x)
    y_mean, y_std = calc_mean_std(y)

    # normalized_features: 512*53*80
    normalized_features = y_std * (x - x_mean) / x_std + y_mean
    return normalized_features


'''
load style image
Return: tensor shape 1*c*h*w, nornalized
'''
# def get_style_data(path):
#     std = [0.229, 0.224, 0.225]     # imageNet std
#     mean = [0.485, 0.456, 0.406]    # imageNet mean

#     transform = tv.transforms.Compose([
#         tv.transforms.ToTensor(),
#         tv.transforms.Normalize(mean, std)
#     ])

#     img = tv.datasets.folder.default_loader('')   # TODO
#     data = transform(img)
#     return data.unsqueeze(0)


'''
Input:  b,c,h,w   0 ~ 255
Output: b,c,h,w  -2 ~ 2
'''
# def normalize_batch(batch):
#     std = [0.229, 0.224, 0.225]     # imageNet std
#     mean = [0.485, 0.456, 0.406]    # imageNet mean
    
#     bt_mean = batch.data.new(mean).view(1, -1, 1, 1)
#     bt_std = batch.data.new(std).view(1, -1, 1, 1)

#     bt_mean = (bt_mean.expand_as(batch.data))
#     bt_std = (bt_std.expand_as(batch.data))

#     out = (batch / 255.0 - mean) / std
#     return out