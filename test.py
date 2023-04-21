import torch as t
import os
import torchnet.meter as meter
import torchvision as tv
from torch.utils.data import DataLoader

from PIL import Image

from configs import DefaultConfig
from models import TransferNet

cfg = DefaultConfig()
device = t.device('cpu' if not t.cuda.is_available() else cfg.device)
IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD=[0.229,0.224,0.225]
mean=t.Tensor(IMAGENET_MEAN).reshape(-1, 1, 1)
std=t.Tensor(IMAGENET_STD).reshape(-1, 1, 1)

# 图片处理
transfroms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
content_image = Image.open(cfg.test_content_path)
content_image = transfroms(content_image)
content_image = content_image.unsqueeze(0).to(device)

style_image = Image.open(cfg.test_style_path)
style_image = transfroms(style_image)
style_image = style_image.unsqueeze(0).to(device)

# 加载模型
net = TransferNet()
model_path = cfg.net_save_root + '/' + cfg.net_path
assert os.path.exists(model_path)
state_dict = t.load(model_path)
net.load_state_dict(state_dict)
net.to(device)
net.eval()

# 风格迁移与保存
output = net.generate(content_image, style_image)
output_data = output.cpu()*std + mean
tv.utils.save_image(output_data.clamp(0,1), cfg.output_dir + '/' + 'result.png')