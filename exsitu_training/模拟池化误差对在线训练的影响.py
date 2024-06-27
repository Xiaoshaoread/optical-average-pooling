""" <shao>
由于池化层的参数不参与网络参数的更新，因此制造误差不会被训练和修复，故单独考虑池化误差对OCNN训练的影响
"""

import sys
from time import time
import torch
from torch import nn, optim
# from exsitu_training.mzi_conv2d_ex_train import MZIConv2d
# from exsitu_training.mzi_linear_ex_train import MZILinear
from torchonn.layers.mzi_conv2d import MZIConv2d
from torchonn.layers.mzi_linear import MZILinear
from torchonn.models.base_model import ONNBaseModel
from torchvision import datasets, transforms
import numpy as np
from torchonn.op.matrix_parametrization import RealUnitaryDecomposerBatch
from torchonn.op.mzi_op import checkerboard_to_vector, vector_to_checkerboard
import torch.nn.functional as F

class ConsoleLogger:
    def __init__(self, filename="D:/研究生工作/资料/小论文资料/测试/结果/phase_training_SVD_std="):
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

for std_pooling in [0.01, 0.1]:

    # 创建一个实例以重定向控制台输出
    console_logger = ConsoleLogger()
    # 在代码运行前调用 save 方法，提供 std 参数
    console_logger.save(std_pooling)
    # 保存原始的 sys.stdout
    original_stdout = sys.stdout
    # 重定向 sys.stdout 到 ConsoleLogger
    sys.stdout = console_logger
    # 超参设置
    num_epochs = 10
    num_classes = 10
    learning_rate = 0.005
    BATCH_SIZE = 100
    DEVICE = torch.device('cpu')
    W = torch.tensor(
        [[1 / 16], [1 / 16], [1 / 16], [1 / 16], [1 / 16], [1 / 16], [1 / 16], [1 / 16], [1 / 16], [1 / 16],
         [1 / 16], [1 / 16], [1 / 16], [1 / 16], [1 / 16], [1 / 16]])
    U_grid, S, V = torch.svd(W, some=False)
    # 获取加噪后的池化核序列，由于仅考虑了U模块噪声，因此需要除以4
    custom_kernel = (add_gaussian_noise_to_phase(U_grid, std_pooling).T)[0] / 4
    # print(custom_kernel)
    # 将池化核序列补充完整进行pytorch的跨步卷积运算来等效池化（8，1，4，4）在conv2d里面（通道，核个数，核尺寸1，核尺寸2）
    custom_kernel = custom_kernel.view(1, 1, 4, 4).expand(8, -1, -1, -1)
    # print(custom_kernel)
    custom_kernel = custom_kernel.float()

    def mnist_loader(train=True, batch_size=BATCH_SIZE, shuffle=False):
        loader = torch.utils.data.DataLoader(
            datasets.MNIST('E:/python/pyCharm/PycharmProjects/MNIST_data', train=train, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=shuffle)
        return loader

    train_loader = mnist_loader(train=True, batch_size=BATCH_SIZE, shuffle=True);
    test_loader = mnist_loader(train=False, batch_size=BATCH_SIZE, shuffle=False)

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
                mode="phase",  # 提供额外的矩阵分解
                decompose_alg="clements",  # 矩阵分解模式
                photodetect=True,  # 光电探测器
                device=device,
            )
            self.pool = nn.AdaptiveAvgPool2d(7)  # 规定池化后输出图片尺寸为(N,C,5,5)

            self.linear = MZILinear(
                in_features=8 * 7 * 7,  # in_features表示输入为上一层输出（N,C,H,W）的后参为参数的乘积
                out_features=10,  # 表示输出的分类的类别数num_classes
                bias=True,
                # miniblock=4,
                mode="phase",
                decompose_alg="clements",
                photodetect=True,
                device=device,
            )
            self.bn = nn.BatchNorm2d(8, affine=True)

        def forward(self, x):
            x = torch.relu((self.bn(self.conv(x))))
            # print(x.shape)
            Y = F.conv2d(x, custom_kernel, stride=4, padding=0, dilation=1, groups=8)
            # print(Y.shape)
            # Y = self.pool(x)
            x = Y.flatten(1)
            # print(x.shape)
            x = torch.relu(self.linear(x))
            return x

    model = ONNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)
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

        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # 将梯度清零
            # 向网络中输入images，得到outputs,在这一步的时候模型会自动调用model.forward(images)函数
            outputs = model(images)  # outputs = Tenser（100,10）,表示这一百张图片各自生成对应数字0-9的概率
            loss = criterion(outputs, labels)  # 将损失向输入侧进行反向传播,计算梯度
            loss.backward()  # loss.backward() 函数，将损失loss向输入侧进行反向传播来计算网络参数的梯度：
            optimizer.step()  # optimizer.step()，通过最小化损失函数loss（利用前面的梯度）来对参数进行更新
            acc_loss += loss.item()  # .item()的作用是取出单元素张量的元素值并返回该值，并且精度更高#这一步是每一小批次的损失loss进行累加
            # 将每一次的loss进行累积
            pred = outputs.argmax(dim=1)  # y=argmax(f(t)),取使得f(t)函数取到最大值时的参数t ，，，，找到每个图片最大概率对应的数字，也即对应的预测值
            acc = (pred == labels).float().mean()  # 计算（预测值pred和结果labels相等）平均准确率，mean()函数为计算平均值
            pre += acc.item()  # item（）,取值且精度更高

            if (batch_idx + 1) % 30 == 0 :
                # torch.save(model.state_dict(), 'Model_pt文件/model(9_10).pt')
                # print("第", batch_idx + 1, "批次后的卷积核权重：", model.conv.weight)
                t = time() - t0
                t0 = time()
                print("Epoch:", '[{}/{}]'.format(epoch + 1, num_epochs), "Step:",
                      '[{}/{}]'.format(batch_idx + 1, total_step),
                      "loss:", loss.item(), "accuracy:", '{:.2%}'.format(acc.item()), "Time:",
                      '{:.4f}'.format(t / 100 * 1e3), "ms")
    with torch.no_grad():  # 在模型中禁用autograd功能，加快计算,预测模式不需要backward()函数来计算梯度·
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # print("epoch =",epoch+1,': {} %'.format((correct / total) * 100))
        print('测试集准确率: {} %'.format((correct / total) * 100))

    # 在代码运行结束后调用 close 方法关闭日志文件
    console_logger.close()
    # 恢复原始的 sys.stdout
    sys.stdout = original_stdout
