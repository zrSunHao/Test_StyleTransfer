import torch.nn as nn

from .std import calc_mean_std

def get_conetnt_loss(content, adain):
    return nn.functional.mse_loss(content, adain)
    
def get_style_loss(content, style):
    loss = 0
    for i, j in zip(content, style):
        content_mean, content_std = calc_mean_std(i)
        style_mean, style_std = calc_mean_std(i)
        loss_mean = nn.functional.mse_loss(content_mean, style_mean)
        loss_std = nn.functional.mse_loss(content_std, style_std)
        loss += loss_mean + loss_std
    return loss