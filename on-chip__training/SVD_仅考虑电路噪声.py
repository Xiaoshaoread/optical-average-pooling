from time import time
import torch
from torch import nn
from in_situ_training.mzi_conv2d import MZIConv2d
from in_situ_training.mzi_linear import MZILinear
from torchonn.models.base_model import ONNBaseModel
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
from torchonn.op.matrix_parametrization import RealUnitaryDecomposerBatch
from torchonn.op.mzi_op import (
    checkerboard_to_vector,
    vector_to_checkerboard,
)
import torch.nn.functional as F

# 超参设置
W=torch.tensor([[1/16],[1/16],[1/16],[1/16],[1/16],[1/16],[1/16],[1/16],[1/16],[1/16],[1/16],[1/16],[1/16],[1/16],[1/16],[1/16]])
U_grid, S, V = torch.svd(W, some=False)

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

import sys

specific_values = [0.001,0.002,0.003]

for phase_noise in specific_values:
    # 定义一个新的类，用于重定向和保存控制台输出
    class ConsoleLogger:
        def __init__(self, filename="C:/Users/14221/Desktop/仅考虑电路噪声/SVD/SVD_仅电路_noise_std="):
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

    std_pooling = 0
    # 创建一个实例以重定向控制台输出
    console_logger = ConsoleLogger()
    # 在代码运行前调用 save 方法，提供 std 参数
    console_logger.save(phase_noise)
    # 保存原始的 sys.stdout
    original_stdout = sys.stdout
    # 重定向 sys.stdout 到 ConsoleLogger
    sys.stdout = console_logger

    # std_pooling = 0

    onn_mat = (add_gaussian_noise_to_phase(U_grid, std_pooling).T)[0] / 4
    # onn_mat = (U_grid.T)[0] / 4

    # grid_U_noisy = (add_gaussian_noise_to_phase(U_grid, std_pooling).T)[0]
    #卷积层和全连接层的初始化在代码里面改
    num_epochs = 10
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
    test_loader = mnist_loader(train=False, batch_size=BATCH_SIZE, shuffle=False)

    # Get X for testing
    for data, target in train_loader:
        # print(target)
        break
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
                reset_phase_std=std_pooling,
                # device = torch.device("cpu"),
            )

            self.linear = MZILinear(
                in_features=8 * 7 * 7,
                out_features=10,
                bias=True,
                # miniblock=4,
                mode="phase",
                decompose_alg="clements",
                photodetect=True,
                device=device,
                reset_phase_std=std_pooling,
            )
            self.bn = nn.BatchNorm2d(8, affine=True)
            self.conv.set_phase_variation(phase_noise)
            self.linear.set_phase_variation(phase_noise)
            # self.conv.reset_parameters()
            # self.linear.reset_parameters()

        def forward(self, x):
            x = torch.relu(self.bn(self.conv(x)))
            # 获取加噪后的池化核序列，由于仅考虑了U模块噪声，因此需要除以4
            # print(custom_kernel)
            # 将池化核序列补充完整进行pytorch的跨步卷积运算来等效池化（8，1，4，4）在conv2d里面（通道，核个数，核尺寸1，核尺寸2）
            custom_kernel = onn_mat.view(1, 1, 4, 4).expand(8, -1, -1, -1)
            # print(custom_kernel)
            custom_kernel = custom_kernel.float()
            Y = F.conv2d(x, custom_kernel, stride=4, padding=0, dilation=1, groups=8)

            x = Y.flatten(1)
            # x = self.linear(x)
            x = torch.relu(self.linear(x))
            return x

    # PART 3：创建一个CNN实例
    model = ONNModel()
    print("conv:",model.conv.phase_noise_std)
    print("Fc:", model.linear.phase_noise_std)
    # # 该函数包含了 SoftMax activation 和 cross entorpy，所以在神经网络结构定义的时候不需要定义softmax activation
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # # 在nn.Module类中，方法 nn.parameters()可以让pytorch追踪所有CNN中需要训练的模型参数，让他知道要优化的参数是哪些
    #  卷积层和全连接层引入相位噪声，标准差为std
    # model.conv.set_phase_variation(std_conv_linear)
    # model.linear.set_phase_variation(std_conv_linear)
    # # PART 4：训练模型

    # #训练数据集长度
    total_step = len(train_loader)
    train_loss = []
    loss_list = []
    acc_list = []
    list = []
    temp = 0

    # acc_list = []
    for epoch in range(num_epochs):
        acc_loss = 0
        pre = 0
        total = 0
        t0 = time()
        #     # 遍历训练数据(images,label)
        for batch_idx, (images, labels) in enumerate(train_loader):
            # print(len(train_loader))
            # 将梯度清零
            # print(labels)  # 打印每一批次即将训练的数据集的标签值
            optimizer.zero_grad()
            # 向网络中输入images，得到outputs,在这一步的时候模型会自动调用model.forward(images)函数
            outputs = model(images)
            # 计算损失（交叉熵算法）
            loss = criterion(outputs, labels)
            # 将损失向输入侧进行反向传播
            loss.backward()
            optimizer.step()  # 在整个批次上计算的梯度的平均值，并据此更新模型参数
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

            if (batch_idx + 1) % 10 == 0:
                t = time() - t0
                t0 = time()

                print("Epoch:", '[{}/{}]'.format(epoch + 1, num_epochs), "Step:",
                      '[{}/{}]'.format(batch_idx + 1, total_step),
                      "loss:", loss.item(), "accuracy:", '{:.2%}'.format(acc.item()), "Time/it:",
                      '{:.4f}'.format(t / 100 * 1e3), "ms")
                # break

    # 切换到测试模式
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        # 循环遍历测试集中的图片，images：图片像素值，labels：图片上是数字几
        for images, labels in test_loader:
            outputs = model(images)  # model为ONN模型，将图片经过网络得到outputs
            _, predicted = torch.max(outputs.data, 1)  # outputs是十个值组成的向量，取值最大的为预测值
            total += labels.size(0)  # 总共输入到网络参与测试的图片数
            correct += (predicted == labels).sum().item()  # 若一张图片识别正确，correct+1
        # 用识别正确的图片数除以图片总数，得到准确率
        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
        # print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i , len(train_loader.dataset),100. * i / len(train_loader), train_loss/ i))
    print("SVD平均池化4×4", 'conv.mode= ', model.conv.mode, ' ,Linear.mode= ', model.linear.mode)

    # 在代码运行结束后调用 close 方法关闭日志文件
    console_logger.close()
    # 恢复原始的 sys.stdout
    sys.stdout = original_stdout

if __name__ == "__main__":
    input_OCNN = ONNModel()  # 类的实例化
    input_OCNN(data)  # 给网络输入图片