import sys
from time import time
import torch
from torch import nn
# from exsitu_training.mzi_conv2d_ex_train import MZIConv2d
# from exsitu_training.mzi_linear_ex_train import MZILinear
from torchonn.layers.mzi_conv2d import MZIConv2d
from torchonn.layers.mzi_linear import MZILinear
from torchonn.models.base_model import ONNBaseModel
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as F
from torchonn.op.matrix_parametrization import RealUnitaryDecomposerBatch
from torchonn.op.mzi_op import checkerboard_to_vector, vector_to_checkerboard

std = 0.003
# 定义一个新的类，用于重定向和保存控制台输出
class ConsoleLogger:
    def __init__(self, filename="E:\python\pyCharm\PycharmProjects\pythonProject2\in_situ_training\离线测试\std="):
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


# 创建一个实例以重定向控制台输出
console_logger = ConsoleLogger()
# 在代码运行前调用 save 方法，提供 std 参数
console_logger.save(std)
# 保存原始的 sys.stdout
original_stdout = sys.stdout
# 重定向 sys.stdout 到 ConsoleLogger
sys.stdout = console_logger

for p_dropout in np.arange(0, 1, 0.05):
    #超参数配置
    t0 = time()
    BATCH_SIZE = 10000


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
                # reset_phase_std=0,  # 用于设置usv模式的制造误差
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
                # reset_phase_std=0,  # 用于设置usv模式的制造误差
                device=device,
            )
            self.bn = nn.BatchNorm2d(8, affine=True)

            self.conv.set_phase_variation(0)
            self.linear.set_phase_variation(0)
            # self.conv.reset_parameters()
            # self.linear.reset_parameters()

        def forward(self, x):
            x = self.pool(torch.relu(self.bn(self.conv(x))))
            x = x.flatten(1)  # 平坦化
            x = F.dropout(x, p=p_dropout)

            x = self.linear(x)
            # x = torch.relu(x)
            # x = torch.softmax(x, dim=1)

            return x
    model = ONNModel()
    # print(model)

    #  卷积层和全连接层引入相位噪声，标准差为std
    # model.conv.set_phase_variation(std)
    # model.linear.set_phase_variation(std)

    # checkpoint = torch.load('MNIST模型.pth文件/usv_usv_4×4平均池化_100epochs.pth')
    checkpoint = torch.load('E:\python\pyCharm\PycharmProjects\pythonProject2\in_situ_training\离线测试\model.pth')
    # print(checkpoint)
    model.load_state_dict(checkpoint)

    def mnist_loader(train=True, batch_size=BATCH_SIZE, shuffle=False):
        loader = torch.utils.data.DataLoader(
            datasets.MNIST('E:/python/pyCharm/PycharmProjects/MNIST_data', train=train, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=shuffle)
        return loader
    test_loader = mnist_loader(train=False, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        t = time() - t0
        print('p = ', p_dropout,',测试准确率为: {} %'.format((correct / total) * 100), '耗时：{:.4f}'.format(t / 100 * 1e3), "ms")



# 在代码运行结束后调用 close 方法关闭日志文件
console_logger.close()
# 恢复原始的 sys.stdout
sys.stdout = original_stdout
