# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 16:28:40 2021

@author: HP
"""
from torchonn.op.mzi_op import project_matrix_to_unitary
from typing import List, Union
from time import time
import torch
from torch import nn
from torch._tensor import Tensor
from torch.types import Device, _size
# from torchonn.layers.mzi_conv2d import MZIBlockConv2d, MZIConv2d
from cw_mzi_conv2d import MZIConv2d
# from torchonn.layers.mzi_linear import MZIBlockLinear, MZILinear
from cw_mzi_linear import MZILinear
from torchonn.models.base_model import ONNBaseModel
from collections import OrderedDict
from torchvision import datasets, transforms
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F


# 超参设置
num_epochs = 100
num_classes = 10
learning_rate = 0.005
BATCH_SIZE = 100
USE_CUDA = torch.cuda.is_available()

DEVICE = torch.device('cpu')

def cifar10_loader(train=True, batch_size=BATCH_SIZE, shuffle=False):
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='E:/python/pyCharm/PycharmProjects/CIFAR10_data', train=train, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])),
        batch_size=batch_size, shuffle=shuffle)

    return loader

# 使用CIFAR-10数据集加载器
train_loader = cifar10_loader(train=True, batch_size=BATCH_SIZE, shuffle=True)
test_loader = cifar10_loader(train=False, batch_size=BATCH_SIZE, shuffle=False)


# Get X for testing
for data, target in train_loader:
    # print(target)
    break

data, target = data.to(DEVICE), target.to(DEVICE)

class ONNModel(ONNBaseModel):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()  # super()函数是用于调用父类(超类)的一个方法。
        self.conv = MZIConv2d(  ###卷积
            in_channels=3,  # 输入通道数
            out_channels=8,  # 输出通道数
            kernel_size=3,  # 卷积核维数
            stride=1,  # 步长
            padding=1,  # 填充
            dilation=1,
            bias=True,
            # miniblock=4,
            mode="usv",  # 提供额外的矩阵分解
            decompose_alg="clements",  # 矩阵分解模式
            photodetect=True,  # 光电探测器
            device=device,
        )
        self.pool = nn.AdaptiveAvgPool2d(7)  # 规定池化后输出图片尺寸为(N,C,5,5)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.linear = MZILinear(  # 级联mzi构造的基于svd的线性层
            in_features=8 * 7 * 7,  # in_features表示输入为上一层输出（N,C,H,W）的后参为参数的乘积
            out_features=num_classes,  # 表示输出的分类的类别数num_classes
            bias=True,
            # miniblock=4,
            mode="usv",
            decompose_alg="clements",
            photodetect=True,
            device=device,
        )
        self.bn = nn.BatchNorm2d(8, affine=True)

        # self.conv.set_phase_variation(0)
        # self.linear.set_phase_variation(0)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        # x = self.pool(torch.relu(self.bn(self.conv(x))))  # bn归一化
        # x = self.pool(torch.nn.functional.softplus(self.bn(self.conv(x))))
        x = x.flatten(1)  # 平坦化
        # print('平坦化后的形状',x.shape)
        x = self.linear(x)
        return x

# PART 3：创建一个CNN实例
model = ONNModel()
# print("创建ONN实例卷积核权重" + str(model.conv.weight))
# print(model.conv.weight)
# print(model)

# # 该函数包含了 SoftMax activation 和 cross entorpy，所以在神经网络结构定义的时候不需要定义softmax activation
criterion = nn.CrossEntropyLoss()
# 调用交叉熵函数，用于计算损失函数
# # 第一个参数:我们想要训练的参数。
# # 在nn.Module类中，方法 nn.parameters()可以让pytorch追踪所有CNN中需要训练的模型参数，让他知道要优化的参数是哪些
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# 优化器对象Optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数。
# print(model.parameters());
# # PART 4：训练模型

# 训练数据集长度
total_step = len(train_loader)
# train_loss = []
list = []
temp = 0
acc1 = []
total_t = 0
weights = []  # 每迭代一次就记录一次卷积核的变化值
weights_batch = []  # 每隔100批次就会记录一下卷积核的变化值
# acc_list = []
for epoch in range(num_epochs):  # 对训练数据进行num_epochs次迭代
    acc_loss = 0
    pre = 0  # 在每次迭代之前都要将上一次迭代得到的累加损失和
    total = 0
    t0 = time()
    #     # 遍历训练数据(images,label)
    for batch_idx, (images, labels) in enumerate(
            train_loader):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        # print(len(train_loader))
        #  打印标签值
        # print(labels)
        # 其中batch_idx从0~599，而(images, labels)为对应序号的训练集内容（包括训练值和结果）
        optimizer.zero_grad()  # 将梯度清零
        # 向网络中输入images，得到outputs,在这一步的时候模型会自动调用model.forward(images)函数
        # print(images.shape)
        outputs = model(images)  # outputs = Tenser（100,10）,表示这一百张图片各自生成对应数字0-9的概率
        # 计算损失（交叉熵算法）
        loss = criterion(outputs, labels)  # 将损失向输入侧进行反向传播,计算梯度
        loss.backward()  # loss.backward() 函数，将损失loss向输入侧进行反向传播来计算网络参数的梯度：

        # 优化器对参数进行更新
        # print("卷积核权重1" + str(model.conv.weight))
        optimizer.step()  # optimizer.step()，通过最小化损失函数loss（利用前面的梯度）来对参数进行更新
        # # print("最终卷积核权重" + str(model.conv.weight))
        #
        acc_loss += loss.item()  # .item()的作用是取出单元素张量的元素值并返回该值，并且精度更高#这一步是每一小批次的损失loss进行累加
        # 将每一次的loss进行累积
        pred = outputs.argmax(dim=1)  # y=argmax(f(t)),取使得f(t)函数取到最大值时的参数t ，，，，找到每个图片最大概率对应的数字，也即对应的预测值
        acc = (pred == labels).float().mean()  # 计算（预测值pred和结果labels相等）平均准确率，mean()函数为计算平均值
        pre += acc.item()  # item（）,取值且精度更高

        if (batch_idx + 1) % 100 == 0:
            # torch.save(model.state_dict(), 'Model_pt文件/model(9_10).pt')
            # print("第", batch_idx + 1, "批次后的卷积核权重：", model.conv.weight)
            t = time() - t0
            t0 = time()
            print("Epoch:", '[{}/{}]'.format(epoch + 1, num_epochs), "Step:",
                  '[{}/{}]'.format(batch_idx + 1, total_step),
                  "loss:", loss.item(), "accuracy:", '{:.2%}'.format(acc.item()), "Time:",
                  '{:.4f}'.format(t / 100 * 1e3), "ms")
            # print(model.conv.weight)
            # print(model.linear.U)
            # print(model.linear.phase_U)

            # print(model.linear.weight.shape)
            # print("第", batch_idx + 1, "批次后的全连接层权重：", model.linear.weight)
    # t = time() - t0
    # t0 = time()
    # print("Epoch:", '[{}/{}]'.format(epoch + 1, num_epochs),
    #       "loss:", loss.item(), "accuracy:", '{:.2%}'.format(acc.item()), "Time:",
    #       '{:.4f}'.format(t / 100 * 1e3), "ms")

    # if acc.item() > temp:
    #     torch.save(model.state_dict(), 'model.pt')
    #     temp = acc.item()

# 下面保存的参数不再进行修改
torch.save(model.state_dict(), 'usv_usv_4×4平均池化_100epochs.pth')

model.eval()
with torch.no_grad():  # 在模型中禁用autograd功能，加快计算,预测模式不需要backward（）函数来计算梯度·
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the test images: {} %'.format((correct / total) * 100))

# print("conv",model.conv.weight)
# print("linear",model.linear.weight)

