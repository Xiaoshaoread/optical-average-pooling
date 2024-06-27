import sys
from time import time
import torch
from torch import nn, optim
from in_situ_training.mzi_conv2d import MZIConv2d
from in_situ_training.mzi_linear import MZILinear
# from torchonn.layers.mzi_conv2d import MZIConv2d
# from torchonn.layers.mzi_linear import MZILinear
from torchonn.models.base_model import ONNBaseModel
from torchvision import datasets, transforms
import numpy as np
from torchonn.op.matrix_parametrization import RealUnitaryDecomposerBatch
from torchonn.op.mzi_op import checkerboard_to_vector, vector_to_checkerboard

for std in np.arange(0.030, 0.031, 0.001):
    # 定义一个新的类，用于重定向和保存控制台输出
    class ConsoleLogger:
        def __init__(self, filename="C:/Users/14221/Desktop/电路噪声研究工作/仅考虑电路噪声/电子池化/初始制造误差对训练的影响/std="):
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

        def save(self, std, average_correct):
            if self.logfile is None:
                # 如果 logfile 为空，则说明这是第一次调用 save 方法，创建新文件
                self.logfile = open(f"{self.filename}{std}_accuracy={average_correct:.4f}.txt", "w")
            else:
                # 否则以追加模式打开文件
                self.logfile = open(f"{self.filename}{std}_accuracy={average_correct:.4f}.txt", "a")

        def close(self):
            if self.logfile is not None:
                self.logfile.close()


    # 创建一个实例以重定向控制台输出
    console_logger = ConsoleLogger()
    # 在代码运行前调用 save 方法，提供 std 参数
    console_logger.save(std,0)
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
                mode="usv",
                decompose_alg="clements",
                photodetect=True,
                reset_phase_std=std,
                # device = torch.device("cpu"),
            )
            self.pool = nn.AdaptiveAvgPool2d(7)  # 规定池化后输出图片尺寸为(N,C,5,5)
            self.linear = MZILinear(
                in_features=8 * 7 * 7,
                out_features=10,
                bias=True,
                # miniblock=4,
                mode="usv",
                decompose_alg="clements",
                photodetect=True,
                reset_phase_std=std,
                device=device,
            )
            self.bn = nn.BatchNorm2d(8, affine=True)

            # self.conv.set_phase_variation(phase_noise)
            # self.linear.set_phase_variation(phase_noise)
            # self.conv.reset_parameters()
            # self.linear.reset_parameters()

        def forward(self, x):
            x = self.pool(torch.relu(self.bn(self.conv(x))))
            x = x.flatten(1)  # 平坦化
            x = torch.relu(self.linear(x))

            return x


    # PART 3：创建一个CNN实例
    model = ONNModel()
    print("conv电路噪声:", model.conv.phase_noise_std)
    print("Fc电路噪声:", model.linear.phase_noise_std)
    # # 该函数包含了 SoftMax activation 和 cross entorpy，所以在神经网络结构定义的时候不需要定义softmax activation
    criterion = nn.CrossEntropyLoss()
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
            if (batch_idx + 1) % 100 == 0:
                t = time() - t0
                t0 = time()
                print("Epoch:", '[{}/{}]'.format(epoch + 1, num_epochs), "Step:",
                      '[{}/{}]'.format(batch_idx + 1, total_step),
                      "loss:", loss.item(), "accuracy:", '{:.2%}'.format(acc.item()), "Time/it:",
                      '{:.4f}'.format(t / 100 * 1e3), "ms")

    # 切换到测试模式
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        average_correct = 0
        # 循环遍历测试集中的图片值，images：图片像素，labels：图片上是数字几
        for images, labels in test_loader:
            outputs = model(images)  # model为ONN模型，将图片经过网络得到outputs
            _, predicted = torch.max(outputs.data, 1)  # outputs是十个值组成的向量，取值最大的为预测值
            total += labels.size(0)  # 总共输入到网络参与测试的图片数
            correct += (predicted == labels).sum().item()  # 若一张图片识别正确，correct+1
        # 用识别正确的图片数除以图片总数，得到准确率
        average_correct = (correct / total) * 100
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(average_correct))
    print("FFT平均池化4×4", 'conv.mode= ', model.conv.mode, ' ,Linear.mode= ', model.linear.mode)

    # 关闭重定向
    sys.stdout = console_logger.terminal
    # 重新调用 save 方法，提供 std 和真实的 average_correct 值
    console_logger.save(std, average_correct)
    # 关闭日志文件
    console_logger.close()
