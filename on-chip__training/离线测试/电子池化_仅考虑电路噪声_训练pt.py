import sys
from time import time
import torch
from torch import Tensor, nn
from torch.types import Device, _size
from in_situ_training.mzi_conv2d import MZIConv2d
from in_situ_training.mzi_linear import MZILinear
from torchonn.models.base_model import ONNBaseModel
from collections import OrderedDict
from torchvision import datasets, transforms
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from pool_2d import pool2d
from math import log2, ceil, pi, cos, sin
from functools import reduce
import torch.nn.functional as F

specific_values = [0.003]

for phase_noise in specific_values:
    # phase_noise = 0.003
    # phase_noise 相位更新后引入的相位误差
    # for phase_noise in np.arange(0, 0.002, 0.001):
    # 定义一个新的类，用于重定向和保存控制台输出C:\Users\14221\Desktop\仅考虑电路噪声\FFT\不同训练算法
    class ConsoleLogger:
        def __init__(self,
                     filename="C:/Users/14221/Desktop/电路噪声研究工作/仅考虑电路噪声/电子池化/参数更新/训练pth_relu_dropout(无)_(8_3_3)_phase_U_Adam0.005_30epoch_std="):
            # 你可以在调用这个类的实例时，提供 std 参数，然后将其拼接到文件名中
            self.filename = filename
            self.terminal = sys.stdout
            self.logfile = None  # 将日志文件的初始化延迟到调用 save 方法时

        def write(self, message):
            # print("wrint")
            self.terminal.write(message)
            if self.logfile is not None:
                self.logfile.write(message)

        def flush(self):
            pass

        def save(self, std):
            # 在调用 save 方法时，根据 std 参数拼接完整的文件名
            filename_with_std = f"{self.filename}{std}.txt"
            self.logfile = open(filename_with_std, "w")
            # print("save")

        def close(self):
            if self.logfile is not None:
                self.logfile.close()


    # 相位初始化误差，制造误差
    std_reset = 0
    # 创建一个实例以重定向控制台输出
    console_logger = ConsoleLogger()
    # 在代码运行前调用 save 方法，提供 std 参数
    console_logger.save(phase_noise)
    # 保存原始的 sys.stdout
    original_stdout = sys.stdout
    # 重定向 sys.stdout 到 ConsoleLogger
    sys.stdout = console_logger

    num_epochs = 30
    num_classes = 10
    learning_rate = 0.005
    BATCH_SIZE = 100
    DEVICE = torch.device('cpu')


    def mnist_loader(train=True, batch_size=BATCH_SIZE, shuffle=False):
        loader = torch.utils.data.DataLoader(
            datasets.MNIST('E:/python/pyCharm/PycharmProjects/MNIST_data', train=train, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=shuffle)

        return loader


    train_loader = mnist_loader(train=True, batch_size=BATCH_SIZE, shuffle=False);
    test_loader = mnist_loader(train=False, batch_size=BATCH_SIZE, shuffle=False);

    # Get X for testing
    for data, target in mnist_loader(train=True, batch_size=1, shuffle=True):
        continue

    data, target = data.to(DEVICE), target.to(DEVICE)


    class ONNModel(ONNBaseModel):
        def __init__(self, device=torch.device("cpu")):
            super().__init__()
            self.conv = MZIConv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True,
                # miniblock=4,
                mode="phase",
                decompose_alg="clements",
                photodetect=True,
                reset_phase_std=std_reset,
                # device = torch.device("cpu"),
            )
            self.pool = nn.AdaptiveAvgPool2d(3)  # 规定池化后输出图片尺寸为(N,C,5,5)
            self.linear = MZILinear(
                in_features=8 * 3 * 3,
                out_features=10,
                bias=True,
                # miniblock=4,
                mode="phase",
                decompose_alg="clements",
                photodetect=True,
                reset_phase_std=std_reset,
                device=device,
            )
            self.bn = nn.BatchNorm2d(8, affine=True)

            self.conv.set_phase_variation(phase_noise)
            self.linear.set_phase_variation(phase_noise)
            # self.conv.reset_parameters()
            # self.linear.reset_parameters()

        def forward(self, x):
            x = self.pool(torch.relu(self.bn(self.conv(x))))
            x = x.flatten(1)  # 平坦化
            # x = F.dropout(x, p=p_dropout, training=self.training)

            x = self.linear(x)
            # x = torch.relu(x)
            # x = torch.softmax(x, dim=1)

            return x


    # PART 3：创建一个CNN实例
    # print()
    model = ONNModel()
    print("lr=", learning_rate, " ,epoch=", num_epochs, ' ,conv.mode= ', model.conv.mode, ' ,Linear.mode= ',
          model.linear.mode)
    print("conv_size=", model.conv.out_channels, "*", model.conv.kernel_size, "Fc_size=", model.linear.in_features)
    print("conv_phase_std=", model.conv.phase_noise_std, " ,Fc_phase_std=", model.linear.phase_noise_std)
    # # 该函数包含了 SoftMax activation 和 cross entorpy，所以在神经网络结构定义的时候不需要定义softmax activation
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    # # 第一个参数:我们想要训练的参数。

    # adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # SGD
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    # # 在nn.Module类中，方法 nn.parameters()可以让pytorch追踪所有CNN中需要训练的模型参数，让他知道要优化的参数是哪些

    # # PART 4：训练模型
    # #训练数据集长度
    total_step = len(train_loader)
    train_loss = []
    loss_list = []
    acc_list = []
    list = []
    temp = 0
    total_t = 0
    # acc_list = []
    for epoch in range(num_epochs):
        acc_loss = 0
        pre = 0
        total = 0
        t0 = time()
        #     # 遍历训练数据(images,label)
        for batch_idx, (images, labels) in enumerate(train_loader):
            # print(len(train_loader))
            # print(labels)
            # 将梯度清零
            optimizer.zero_grad()
            # 向网络中输入images，得到outputs,在这一步的时候模型会自动调用model.forward(images)函数
            outputs = model(images)
            # 计算损失（交叉熵算法）
            loss = criterion(outputs, labels)
            # 将损失向输入侧进行反向传播
            loss.backward()
            # print(loss.item())
            optimizer.step()
            # 优化器对梯度进行更新
            acc_loss += loss.item()
            loss_list.append(loss.item())
            # 将每一次的loss进行累积
            pred = outputs.argmax(dim=1)
            acc = (pred == labels).float().mean()
            pre += acc.item()
            acc_list.append(acc.item())
            # 将每一次识别的准确率进行累积
            # "Step:",'[{}/{}]'.format(batch_idx + 1, total_step),
            if (batch_idx + 1) % 100 == 0 and batch_idx != 0:
                t = time() - t0
                t0 = time()
                total_t += t
                print("Epoch:", '[{}/{}]'.format(epoch + 1, num_epochs),
                      # model.conv.U[0][0].detach().numpy(),
                      # model.linear.U[0][0].detach().numpy(),
                      model.conv.phase_U[0].numpy(),
                      model.linear.phase_U[0].numpy(),
                      "Step:", '[{}/{}]'.format(batch_idx + 1, total_step),
                      "loss:", loss.item(), "accuracy:", '{:.2%}'.format(acc.item()),
                      "Time:", '{:.4f}'.format(t / 100 * 1e3), "ms")
                # print("Epoch:", '[{}/{}]'.format(epoch + 1, num_epochs), model.conv.phase_U[0].numpy(), "Step:",
                #       '[{}/{}]'.format(batch_idx + 1, total_step),
                #       "loss:", loss.item(), "accuracy:", '{:.2%}'.format(acc.item()), "Time/it:",
                #       '{:.4f}'.format(t / 100 * 1e3), "ms")
            # if (batch_idx + 1) % 300 == 0:
            #     break;
        # 指定要保存的文件名
        # file_path = './'
        # 使用torch.save()保存模型状态
        torch.save(model.state_dict(), 'model.pth')
    print('The total time taken to train the model：' + '{:.4f}'.format(total_t / 100 * 1e3) + "ms")
    # 切换到测试模式
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        # 循环遍历测试集中的图片值，images：图片像素，labels：图片上是数字几
        for images, labels in test_loader:
            outputs = model(images)  # model为ONN模型，将图片经过网络得到outputs
            _, predicted = torch.max(outputs.data, 1)  # outputs是十个值组成的向量，取值最大的为预测值
            total += labels.size(0)  # 总共输入到网络参与测试的图片数
            correct += (predicted == labels).sum().item()  # 若一张图片识别正确，correct+1
        # 用识别正确的图片数除以图片总数，得到准确率
        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
    # print("电子平均池化4×4", 'conv.mode= ', model.conv.mode, ' ,Linear.mode= ', model.linear.mode)
    # 在代码运行结束后调用 close 方法关闭日志文件
    console_logger.close()
    # 恢复原始的 sys.stdout
    sys.stdout = original_stdout
if __name__ == "__main__":
    input_OCNN = ONNModel()  # 类的实例化
    input_OCNN(data)  # 给网络输入图片
