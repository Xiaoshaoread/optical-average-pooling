import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# 定义cifar10_loader函数
def cifar10_loader(train=True, batch_size=64, shuffle=False):
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='E:/python/pyCharm/PycharmProjects/CIFAR10_data', train=train, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])),
        batch_size=batch_size, shuffle=shuffle)

    return loader

# 使用CIFAR-10数据集加载器
train_loader = cifar10_loader(train=True, batch_size=32, shuffle=False)

# 创建保存图片的目录
save_dir = './raw'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 保存每个类别的第一张图片
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class_images = {class_name: None for class_name in classes}

for batch_idx, (data, target) in enumerate(train_loader):
    for i, label in enumerate(target):
        class_name = classes[label]
        if class_images[class_name] is None:
            image = data[i].numpy()  # 转换为numpy数组
            image = (image * 0.5 + 0.5) * 255  # 反归一化
            image = image.transpose(1, 2, 0).astype('uint8')  # 调整通道顺序
            image = Image.fromarray(image)  # 转换为PIL图像
            image.save(os.path.join(save_dir, f'{class_name}.png'))  # 保存图片
            class_images[class_name] = image
            if all(class_images.values()):
                break
    if all(class_images.values()):
        break
