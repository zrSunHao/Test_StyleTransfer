class DefaultConfig(object):
    device = 'cuda'                         # 设备

    data_root = 'D:/WorkSpace/DataSet'      # 训练集根目录
    content_dir = 'COCO_2017_val/'              # 内容图像目录
    style_dir = 'painter_2/'                          # 风格图像目录

    image_size = 256                        # 图像尺寸
    batch_size = 4                         # 每批次图像数量
    num_workers = 0                         # 进程数
    lr = 5e-5                               # 学习率
    print_every = 200                       # 进度输出频率

    epoch_max = 40                          # 训练轮次
    epoch_current = 1                       # 当前轮次
    net_save_root = './checkpoints'         # 模型参数保存目录
    net_path = 'style_40.pth'                 # 训练好的模型

    test_content_path = './test/input/000000007108.jpg'       # 测试用的内容图片
    test_style_path = './test/output/2133.jpg'         # 测试用的风格图片
    output_dir = './test/output'                    # 测试结果输出目录