"""<shao>
改进在于：1、使用跨步卷积代替平均池化，pool ~ conv2d
        2、指定卷积函数权重，即指定池化核大小
"""
import sys
from time import time
import torch
from torch import nn
from exsitu_training.mzi_conv2d_ex_train import MZIConv2d
from exsitu_training.mzi_linear_ex_train import MZILinear
# from torchonn.layers.mzi_conv2d import MZIConv2d
# from torchonn.layers.mzi_linear import MZILinear
from torchonn.models.base_model import ONNBaseModel
from torchvision import datasets, transforms
import numpy as np
from torchonn.op.matrix_parametrization import RealUnitaryDecomposerBatch
from torchonn.op.mzi_op import checkerboard_to_vector, vector_to_checkerboard
import torch.nn.functional as F

for std_all in np.arange(0.005, 0.1, 0.005):
    # 定义一个新的类，用于重定向和保存控制台输出
    class ConsoleLogger:
        def __init__(self, filename="D:/研究生工作/资料/小论文资料/测试/结果/SVD_std="):
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
    console_logger.save(std_all)
    # 保存原始的 sys.stdout
    original_stdout = sys.stdout
    # 重定向 sys.stdout 到 ConsoleLogger
    sys.stdout = console_logger

    num = 50
    correct_sum = 0
    std_pooling = std_all
    std = 0
    for i in range(num):
        #超参数配置
        t0 = time()
        BATCH_SIZE = 10000

        class ONNModel(ONNBaseModel):
            def __init__(self, device=torch.device("cpu")):
                super().__init__()  # super()函数是用于调用父类(超类)的一个方法。
                self.conv = MZIConv2d(
                    in_channels=1,  # 输入通道数
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
                # self.pool = nn.AdaptiveAvgPool2d(7)  # 规定池化后输出图片尺寸为(N,C,5,5)

                self.linear = MZILinear(
                    in_features=8 * 7 * 7,  # in_features表示输入为上一层输出（N,C,H,W）的后参为参数的乘积
                    out_features=10,  # 表示输出的分类的类别数num_classes
                    bias=True,
                    # miniblock=4,
                    mode="usv",
                    decompose_alg="clements",
                    photodetect=True,
                    device=device,
                )
                self.bn = nn.BatchNorm2d(8, affine=True)

            def forward(self, x):
                x = torch.relu((self.bn(self.conv(x))))
                # print(x.shape)
                # 获取加噪后的池化核序列，由于仅考虑了U模块噪声，因此需要除以4
                custom_kernel = (add_gaussian_noise_to_phase(U_grid, std_pooling).T)[0] / 4
                # print(custom_kernel)
                # 将池化核序列补充完整进行pytorch的跨步卷积运算来等效池化（8，1，4，4）在conv2d里面（通道，核个数，核尺寸1，核尺寸2）
                custom_kernel = custom_kernel.view(1, 1, 4, 4).expand(8, -1, -1, -1)
                # print(custom_kernel)
                custom_kernel = custom_kernel.float()
                Y = F.conv2d(x, custom_kernel, stride=4, padding=0, dilation=1, groups=8)
                x = Y.flatten(1)
                # print(x.shape)
                x = torch.relu(self.linear(x))
                return x

        def add_gaussian_noise_to_phase(matrix, std):
            decomposer = RealUnitaryDecomposerBatch(alg="clements")
            delta_list, phi_mat = decomposer.decompose(matrix)
            phase_U = checkerboard_to_vector(phi_mat)

            #  引入高斯噪声
            noise = np.random.normal(loc=0, scale=std, size=phase_U.shape)

            noisy_phase_U = phase_U + noise
            noisy_phi_mat = vector_to_checkerboard(noisy_phase_U)

            noisy_matrix = decomposer.reconstruct(delta_list, noisy_phi_mat)
            return noisy_matrix

        #  定义池化核序列
        W = torch.tensor(
                        [[1 / 16], [1 / 16], [1 / 16], [1 / 16], [1 / 16], [1 / 16], [1 / 16], [1 / 16], [1 / 16], [1 / 16],
                         [1 / 16], [1 / 16], [1 / 16], [1 / 16], [1 / 16], [1 / 16]])
        U_grid, S, V = torch.svd(W, some=False)

        model = ONNModel()
        # print(model)

        #  卷积层和全连接层引入相位噪声，标准差为std
        model.conv.set_phase_variation(std)
        model.linear.set_phase_variation(std)
        #  对池化层MZI相位引入噪声

        # checkpoint = torch.load('MNIST模型.pth文件/usv_usv_4×4平均池化_100epochs.pth')
        checkpoint = torch.load('E:/python/pyCharm/PycharmProjects/pythonProject2/MNIST模型.pth文件/usv_usv_4×4平均池化_100epochs.pth')
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
            correct_sum += correct

            print('第', i,'次测试准确率为: {} %'.format((correct / total) * 100), '耗时：{:.4f}'.format(t / 100 * 1e3), "ms")

    average_correct = (correct_sum / (num * 100))
    print("std =", std_all, "时，测试集上的准确率为:", average_correct)

    # 在代码运行结束后调用 close 方法关闭日志文件
    console_logger.close()
    # 恢复原始的 sys.stdout
    sys.stdout = original_stdout
