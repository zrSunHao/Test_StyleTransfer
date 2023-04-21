import torch as t
import os
import torchnet.meter as meter
import torchvision as tv
from torch.utils.data import DataLoader

from configs import DefaultConfig
from models import TransferNet

cfg = DefaultConfig()
device = t.device('cpu' if not t.cuda.is_available() else cfg.device)

IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD=[0.229,0.224,0.225]

# 实例化网络模型
net = TransferNet()
model_path = cfg.net_save_root + cfg.net_path
if os.path.exists(model_path):
    state_dict = t.load(model_path)
    net.load_state_dict(state_dict)
net.to(device)

# 数据加载
transforms = tv.transforms.Compose([
    tv.transforms.Resize(cfg.image_size),
    tv.transforms.RandomCrop(cfg.image_size),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
content_dataset = tv.datasets.ImageFolder(cfg.data_root + '/' + cfg.content_dir, 
                                          transforms)
content_dataloader = DataLoader(content_dataset, 
                                cfg.batch_size,
                                shuffle=True,drop_last=True)
style_dataset = tv.datasets.ImageFolder(cfg.data_root + '/' + cfg.style_dir, 
                                        transforms)
style_dataloader = DataLoader(style_dataset, 
                              cfg.batch_size,
                              shuffle=True,drop_last=True)

# 优化器
optimizer = t.optim.Adam(net.parameters(), cfg.lr)

# 损失记录
loss_meter = meter.AverageValueMeter()
dataloader = zip(content_dataloader, style_dataloader)

for epoch in range(cfg.epoch_max):
    loss_meter.reset()
    if((epoch+1)<cfg.epoch_current):
        continue
    net.train()
    step = 1
    for ii, data in enumerate(dataloader):
        optimizer.zero_grad()
        content = data[0][0].to(device)
        style = data[1][0].to(device)

        loss = net(content, style)
        loss.backward()
        optimizer.step()
        loss_meter.add(loss.item())

        if (ii+1) % cfg.print_every == 0:
            loss_print = loss_meter.value()[0]
            print('epoch:{}  num:{} loss:{}'.format(str(epoch+1).ljust(3,' '), 
                                                    str(step*cfg.print_every).ljust(5,' '), 
                                                    str(loss_print).ljust(10,' ')))
            step += 1
    t.save(net.state_dict(), cfg.net_save_root + '/style_{}.pth'.format(epoch+1))
    print('\n')





