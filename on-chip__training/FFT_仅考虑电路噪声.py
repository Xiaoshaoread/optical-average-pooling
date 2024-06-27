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

# 超参设置
# 内置相位theta
theta_A = np.array([pi / 2, pi / 2, pi / 2, pi / 2, pi / 2, pi / 2, pi / 2, pi / 2])
# 第4列MZI相位,外置相位phi的相位值存储在array_A
array_A = np.array(
    [1j * 0, 1j * pi / 8, 1j * 2 * pi / 8, 1j * 3 * pi / 8, 1j * 4 * pi / 8, 1j * 5 * pi / 8, 1j * 6 * pi / 8,
     1j * 7 * pi / 8])
A = np.array(1j ) * [
    [np.exp(array_A[0] + 1j * theta_A[0] / 2) * sin(theta_A[0] / 2), 0, 0, 0, 0, 0, 0, 0, cos(theta_A[0] / 2) * np.exp(1j * theta_A[0] / 2), 0, 0, 0, 0, 0, 0, 0],
    [0, np.exp(array_A[4] + 1j * theta_A[4] / 2) * sin(theta_A[4] / 2), 0, 0, 0, 0, 0, 0, 0, cos(theta_A[4] / 2) * np.exp(1j * theta_A[4] / 2), 0, 0, 0, 0, 0, 0],
    [0, 0, np.exp(array_A[2] + 1j * theta_A[2] / 2) * sin(theta_A[2] / 2), 0, 0, 0, 0, 0, 0, 0, cos(theta_A[2] / 2) * np.exp(1j * theta_A[2] / 2), 0, 0, 0, 0, 0],
    [0, 0, 0, np.exp(array_A[6] + 1j * theta_A[6] / 2) * sin(theta_A[6] / 2), 0, 0, 0, 0, 0, 0, 0, cos(theta_A[6] / 2) * np.exp(1j * theta_A[6] / 2), 0, 0, 0, 0],
    [0, 0, 0, 0, np.exp(array_A[1] + 1j * theta_A[1] / 2) * sin(theta_A[1] / 2), 0, 0, 0, 0, 0, 0, 0, cos(theta_A[1] / 2) * np.exp(1j * theta_A[1] / 2), 0, 0, 0],
    [0, 0, 0, 0, 0, np.exp(array_A[5] + 1j * theta_A[5] / 2) * sin(theta_A[5] / 2), 0, 0, 0, 0, 0, 0, 0, cos(theta_A[5] / 2) * np.exp(1j * theta_A[5] / 2), 0, 0],
    [0, 0, 0, 0, 0, 0, np.exp(array_A[3] + 1j * theta_A[3] / 2) * sin(theta_A[3] / 2), 0, 0, 0, 0, 0, 0, 0, cos(theta_A[3] / 2) * np.exp(1j * theta_A[3] / 2), 0],
    [0, 0, 0, 0, 0, 0, 0, np.exp(array_A[7] + 1j * theta_A[7] / 2) * sin(theta_A[7] / 2), 0, 0, 0, 0, 0, 0, 0, cos(theta_A[7] / 2) * np.exp(1j * theta_A[7] / 2)],
    [np.exp(array_A[0] + 1j * theta_A[0] / 2) * cos(theta_A[0] / 2), 0, 0, 0, 0, 0, 0, 0, -sin(theta_A[0] / 2) * np.exp(1j * theta_A[0] / 2), 0, 0, 0, 0, 0, 0, 0],
    [0, np.exp(array_A[4] + 1j * theta_A[4] / 2) * cos(theta_A[4] / 2), 0, 0, 0, 0, 0, 0, 0, -sin(theta_A[4] / 2) * np.exp(1j * theta_A[4] / 2), 0, 0, 0, 0, 0, 0],
    [0, 0, np.exp(array_A[2] + 1j * theta_A[2] / 2) * cos(theta_A[2] / 2), 0, 0, 0, 0, 0, 0, 0, -sin(theta_A[2] / 2) * np.exp(1j * theta_A[2] / 2), 0, 0, 0, 0, 0],
    [0, 0, 0, np.exp(array_A[6] + 1j * theta_A[6] / 2) * cos(theta_A[6] / 2), 0, 0, 0, 0, 0, 0, 0, -sin(theta_A[6] / 2) * np.exp(1j * theta_A[6] / 2), 0, 0, 0, 0],
    [0, 0, 0, 0, np.exp(array_A[1] + 1j * theta_A[1] / 2) * cos(theta_A[1] / 2), 0, 0, 0, 0, 0, 0, 0, -sin(theta_A[1] / 2) * np.exp(1j * theta_A[1] / 2), 0, 0, 0],
    [0, 0, 0, 0, 0, np.exp(array_A[5] + 1j * theta_A[5] / 2) * cos(theta_A[5] / 2), 0, 0, 0, 0, 0, 0, 0, -sin(theta_A[5] / 2) * np.exp(1j * theta_A[5] / 2), 0, 0],
    [0, 0, 0, 0, 0, 0, np.exp(array_A[3] + 1j * theta_A[3] / 2) * cos(theta_A[3] / 2), 0, 0, 0, 0, 0, 0, 0, -sin(theta_A[3] / 2) * np.exp(1j * theta_A[3] / 2), 0],
    [0, 0, 0, 0, 0, 0, 0, np.exp(array_A[7] + 1j * theta_A[7] / 2) * cos(theta_A[7] / 2), 0, 0, 0, 0, 0, 0, 0, -sin(theta_A[7] / 2) * np.exp(1j * theta_A[7] / 2)]]

# 第3列MZI相位,外置相位phi的相位值存储在array_B
theta_B = np.array([pi / 2, pi / 2, pi / 2, pi / 2, pi / 2, pi / 2, pi / 2, pi / 2])
array_B = np.array([1j * 0, 1j * pi / 4, 1j * 2 * pi / 4, 1j * 3 * pi / 4, 1j * 0, 1j * pi / 4, 1j * 2 * pi / 4, 1j * 3 * pi / 4])
B = np.array(1j) * [
    [np.exp(array_B[0] + 1j * theta_B[0] / 2) * sin(theta_B[0] / 2), 0, 0, 0, cos(theta_B[0] / 2) * np.exp(1j * theta_B[0] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, np.exp(array_B[2] + 1j * theta_B[2] / 2) * sin(theta_B[2] / 2), 0, 0, 0, cos(theta_B[2] / 2) * np.exp(1j * theta_B[2] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, np.exp(array_B[1] + 1j * theta_B[1] / 2) * sin(theta_B[1] / 2), 0, 0, 0, cos(theta_B[1] / 2) * np.exp(1j * theta_B[1] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, np.exp(array_B[3] + 1j * theta_B[3] / 2) * sin(theta_B[3] / 2), 0, 0, 0, cos(theta_B[3] / 2) * np.exp(1j * theta_B[3] / 2), 0, 0, 0, 0, 0, 0, 0, 0],
    [np.exp(array_B[0] + 1j * theta_B[0] / 2) * cos(theta_B[0] / 2), 0, 0, 0, -sin(theta_B[0] / 2) * np.exp(1j * theta_B[0] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, np.exp(array_B[2] + 1j * theta_B[2] / 2) * cos(theta_B[2] / 2), 0, 0, 0, -sin(theta_B[2] / 2) * np.exp(1j * theta_B[2] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, np.exp(array_B[1] + 1j * theta_B[1] / 2) * cos(theta_B[1] / 2), 0, 0, 0, -sin(theta_B[1] / 2) * np.exp(1j * theta_B[1] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, np.exp(array_B[3] + 1j * theta_B[3] / 2) * cos(theta_B[3] / 2), 0, 0, 0, -sin(theta_B[3] / 2) * np.exp(1j * theta_B[3] / 2), 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_B[4] + 1j * theta_B[4] / 2) * sin(theta_B[4] / 2), 0, 0, 0, cos(theta_B[4] / 2) * np.exp(1j * theta_B[4] / 2), 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_B[5] + 1j * theta_B[5] / 2) * sin(theta_B[5] / 2), 0, 0, 0, cos(theta_B[5] / 2) * np.exp(1j * theta_B[5] / 2), 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_B[6] + 1j * theta_B[6] / 2) * sin(theta_B[6] / 2), 0, 0, 0, cos(theta_B[6] / 2) * np.exp(1j * theta_B[6] / 2), 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_B[7] + 1j * theta_B[7] / 2) * sin(theta_B[7] / 2), 0, 0, 0, cos(theta_B[7] / 2) * np.exp(1j * theta_B[7] / 2)],
    [0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_B[4] + 1j * theta_B[4] / 2) * cos(theta_B[4] / 2), 0, 0, 0, -sin(theta_B[4] / 2) * np.exp(1j * theta_B[4] / 2), 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_B[5] + 1j * theta_B[5] / 2) * cos(theta_B[5] / 2), 0, 0, 0, -sin(theta_B[5] / 2) * np.exp(1j * theta_B[5] / 2), 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_B[6] + 1j * theta_B[6] / 2) * cos(theta_B[6] / 2), 0, 0, 0, -sin(theta_B[6] / 2) * np.exp(1j * theta_B[6] / 2), 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_B[7] + 1j * theta_B[7] / 2) * cos(theta_B[7] / 2), 0, 0, 0, -sin(theta_B[7] / 2) * np.exp(1j * theta_B[7] / 2)]]

# 第2列MZI相位,外置相位phi的相位值存储在array_C
theta_C = np.array([pi / 2, pi / 2, pi / 2, pi / 2, pi / 2, pi / 2, pi / 2, pi / 2])
array_C = np.array([1j * 0, 1j * pi / 2, 1j * 0, 1j * pi / 2, 1j * 0, 1j * pi / 2, 1j * 0, 1j * pi / 2])
C = np.array(1j) * [
    [np.exp(array_C[0] + 1j * theta_C[0] / 2) * sin(theta_C[0] / 2), 0, cos(theta_C[0] / 2) * np.exp(1j * theta_C[0] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, np.exp(array_C[1] + 1j * theta_C[1] / 2) * sin(theta_C[1] / 2), 0, cos(theta_C[1] / 2) * np.exp(1j * theta_C[1] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [np.exp(array_C[0] + 1j * theta_C[0] / 2) * cos(theta_C[0] / 2), 0, -sin(theta_C[0] / 2) * np.exp(1j * theta_C[0] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, np.exp(array_C[1] + 1j * theta_C[1] / 2) * cos(theta_C[1] / 2), 0, -sin(theta_C[1] / 2) * np.exp(1j * theta_C[1] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, np.exp(array_C[2] + 1j * theta_C[2] / 2) * sin(theta_C[2] / 2), 0, cos(theta_C[2] / 2) * np.exp(1j * theta_C[2] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, np.exp(array_C[3] + 1j * theta_C[3] / 2) * sin(theta_C[3] / 2), 0, cos(theta_C[3] / 2) * np.exp(1j * theta_C[3] / 2), 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, np.exp(array_C[2] + 1j * theta_C[2] / 2) * cos(theta_C[2] / 2), 0, -sin(theta_C[2] / 2) * np.exp(1j * theta_C[2] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, np.exp(array_C[3] + 1j * theta_C[3] / 2) * cos(theta_C[3] / 2), 0, -sin(theta_C[3] / 2) * np.exp(1j * theta_C[3] / 2), 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_C[4] + 1j * theta_C[4] / 2) * sin(theta_C[4] / 2), 0, cos(theta_C[4] / 2) * np.exp(1j * theta_C[4] / 2), 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_C[5] + 1j * theta_C[5] / 2) * sin(theta_C[5] / 2), 0, cos(theta_C[5] / 2) * np.exp(1j * theta_C[5] / 2), 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_C[4] + 1j * theta_C[4] / 2) * cos(theta_C[4] / 2), 0, -sin(theta_C[4] / 2) * np.exp(1j * theta_C[4] / 2), 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_C[5] + 1j * theta_C[5] / 2) * cos(theta_C[5] / 2), 0, -sin(theta_C[5] / 2) * np.exp(1j * theta_C[5] / 2), 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_C[6] + 1j * theta_C[6] / 2) * sin(theta_C[6] / 2), 0, cos(theta_C[6] / 2) * np.exp(1j * theta_C[6] / 2), 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_C[7] + 1j * theta_C[7] / 2) * sin(theta_C[7] / 2), 0, cos(theta_C[7] / 2) * np.exp(1j * theta_C[7] / 2)],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_C[6] + 1j * theta_C[6] / 2) * cos(theta_C[6] / 2), 0, -sin(theta_C[6] / 2) * np.exp(1j * theta_C[6] / 2), 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_C[7] + 1j * theta_C[7] / 2) * cos(theta_C[7] / 2), 0, -sin(theta_C[7] / 2) * np.exp(1j * theta_C[7] / 2)]]

# 第1列MZI相位,外置相位phi的相位值存储在array_D
theta_D = np.array([pi / 2, pi / 2, pi / 2, pi / 2, pi / 2, pi / 2, pi / 2, pi / 2])
array_D = np.array([1j * 0, 1j * 0, 1j * 0, 1j * 0, 1j * 0, 1j * 0, 1j * 0, 1j * 0])
D = np.array(1j) * [
    [np.exp(array_D[0] + 1j * theta_D[0] / 2) * sin(theta_D[0] / 2), cos(theta_D[0] / 2) * np.exp(1j * theta_D[0] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [np.exp(array_D[0] + 1j * theta_D[0] / 2) * cos(theta_D[0] / 2), -sin(theta_D[0] / 2) * np.exp(1j * theta_D[0] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, np.exp(array_D[1] + 1j * theta_D[1] / 2) * sin(theta_D[1] / 2), cos(theta_D[1] / 2) * np.exp(1j * theta_D[1] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, np.exp(array_D[1] + 1j * theta_D[1] / 2) * cos(theta_D[1] / 2), -sin(theta_D[1] / 2) * np.exp(1j * theta_D[1] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, np.exp(array_D[2] + 1j * theta_D[2] / 2) * sin(theta_D[2] / 2), cos(theta_D[2] / 2) * np.exp(1j * theta_D[2] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, np.exp(array_D[2] + 1j * theta_D[2] / 2) * cos(theta_D[2] / 2), -sin(theta_D[2] / 2) * np.exp(1j * theta_D[2] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, np.exp(array_D[3] + 1j * theta_D[3] / 2) * sin(theta_D[3] / 2), cos(theta_D[3] / 2) * np.exp(1j * theta_D[3] / 2), 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, np.exp(array_D[3] + 1j * theta_D[3] / 2) * cos(theta_D[3] / 2), -sin(theta_D[3] / 2) * np.exp(1j * theta_D[3] / 2), 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_D[4] + 1j * theta_D[4] / 2) * sin(theta_D[4] / 2), cos(theta_D[4] / 2) * np.exp(1j * theta_D[4] / 2), 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_D[4] + 1j * theta_D[4] / 2) * cos(theta_D[4] / 2), -sin(theta_D[4] / 2) * np.exp(1j * theta_D[4] / 2), 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_D[5] + 1j * theta_D[5] / 2) * sin(theta_D[5] / 2), cos(theta_D[5] / 2) * np.exp(1j * theta_D[5] / 2), 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_D[5] + 1j * theta_D[5] / 2) * cos(theta_D[5] / 2), -sin(theta_D[5] / 2) * np.exp(1j * theta_D[5] / 2), 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_D[6] + 1j * theta_D[6] / 2) * sin(theta_D[6] / 2), cos(theta_D[6] / 2) * np.exp(1j * theta_D[6] / 2), 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_D[6] + 1j * theta_D[6] / 2) * cos(theta_D[6] / 2), -sin(theta_D[6] / 2) * np.exp(1j * theta_D[6] / 2), 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_D[7] + 1j * theta_D[7] / 2) * sin(theta_D[7] / 2), cos(theta_D[7] / 2) * np.exp(1j * theta_D[7] / 2)],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(array_D[7] + 1j * theta_D[7] / 2) * cos(theta_D[7] / 2), -sin(theta_D[7] / 2) * np.exp(1j * theta_D[7] / 2)]]

U_fft = reduce(np.dot, [A, B, C, D])
# print('U_fft[0]', U_fft[0])

# 引入高斯噪声

specific_values = [0, 0.001, 0.002, 0.003]

for phase_noise in specific_values:
    # 定义一个新的类，用于重定向和保存控制台输出C:\Users\14221\Desktop\仅考虑电路噪声\FFT\不同训练算法
    class ConsoleLogger:
        def __init__(self, filename="C:/Users/14221/Desktop/仅考虑电路噪声/FFT/不同训练算法/adam_lr_0.001_FFT_仅电路_noise_std="):
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

    def add_gaussian_noise_to_phase(matrix, mean=0, std=std_pooling):
        noise = np.random.normal(loc=mean, scale=std, size=matrix.shape)
        noisy_matrix = matrix + noise
        return noisy_matrix

    noisy_array_A = add_gaussian_noise_to_phase(array_A)
    noisy_theta_A = add_gaussian_noise_to_phase(theta_A)

    noisy_array_B = add_gaussian_noise_to_phase(array_B)
    noisy_theta_B = add_gaussian_noise_to_phase(theta_B)

    noisy_array_C = add_gaussian_noise_to_phase(array_C)
    noisy_theta_C = add_gaussian_noise_to_phase(theta_C)

    noisy_array_D = add_gaussian_noise_to_phase(array_D)
    noisy_theta_D = add_gaussian_noise_to_phase(theta_D)

    noisy_A = np.array(1j ) * [
        [np.exp(noisy_array_A[0] + 1j * noisy_theta_A[0] / 2) * sin(noisy_theta_A[0] / 2), 0, 0, 0, 0, 0, 0, 0, cos(noisy_theta_A[0] / 2) * np.exp(1j * noisy_theta_A[0] / 2), 0, 0, 0, 0, 0, 0, 0],
        [0, np.exp(noisy_array_A[4] + 1j * noisy_theta_A[4] / 2) * sin(noisy_theta_A[4] / 2), 0, 0, 0, 0, 0, 0, 0, cos(noisy_theta_A[4] / 2) * np.exp(1j * noisy_theta_A[4] / 2), 0, 0, 0, 0, 0, 0],
        [0, 0, np.exp(noisy_array_A[2] + 1j * noisy_theta_A[2] / 2) * sin(noisy_theta_A[2] / 2), 0, 0, 0, 0, 0, 0, 0, cos(noisy_theta_A[2] / 2) * np.exp(1j * noisy_theta_A[2] / 2), 0, 0, 0, 0, 0],
        [0, 0, 0, np.exp(noisy_array_A[6] + 1j * noisy_theta_A[6] / 2) * sin(noisy_theta_A[6] / 2), 0, 0, 0, 0, 0, 0, 0, cos(noisy_theta_A[6] / 2) * np.exp(1j * noisy_theta_A[6] / 2), 0, 0, 0, 0],
        [0, 0, 0, 0, np.exp(noisy_array_A[1] + 1j * noisy_theta_A[1] / 2) * sin(noisy_theta_A[1] / 2), 0, 0, 0, 0, 0, 0, 0, cos(noisy_theta_A[1] / 2) * np.exp(1j * noisy_theta_A[1] / 2), 0, 0, 0],
        [0, 0, 0, 0, 0, np.exp(noisy_array_A[5] + 1j * noisy_theta_A[5] / 2) * sin(noisy_theta_A[5] / 2), 0, 0, 0, 0, 0, 0, 0, cos(noisy_theta_A[5] / 2) * np.exp(1j * noisy_theta_A[5] / 2), 0, 0],
        [0, 0, 0, 0, 0, 0, np.exp(noisy_array_A[3] + 1j * noisy_theta_A[3] / 2) * sin(noisy_theta_A[3] / 2), 0, 0, 0, 0, 0, 0, 0, cos(noisy_theta_A[3] / 2) * np.exp(1j * noisy_theta_A[3] / 2), 0],
        [0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_A[7] + 1j * noisy_theta_A[7] / 2) * sin(noisy_theta_A[7] / 2), 0, 0, 0, 0, 0, 0, 0, cos(noisy_theta_A[7] / 2) * np.exp(1j * noisy_theta_A[7] / 2)],
        [np.exp(noisy_array_A[0] + 1j * noisy_theta_A[0] / 2) * cos(noisy_theta_A[0] / 2), 0, 0, 0, 0, 0, 0, 0, -sin(noisy_theta_A[0] / 2) * np.exp(1j * noisy_theta_A[0] / 2), 0, 0, 0, 0, 0, 0, 0],
        [0, np.exp(noisy_array_A[4] + 1j * noisy_theta_A[4] / 2) * cos(noisy_theta_A[4] / 2), 0, 0, 0, 0, 0, 0, 0, -sin(noisy_theta_A[4] / 2) * np.exp(1j * noisy_theta_A[4] / 2), 0, 0, 0, 0, 0, 0],
        [0, 0, np.exp(noisy_array_A[2] + 1j * noisy_theta_A[2] / 2) * cos(noisy_theta_A[2] / 2), 0, 0, 0, 0, 0, 0, 0, -sin(noisy_theta_A[2] / 2) * np.exp(1j * noisy_theta_A[2] / 2), 0, 0, 0, 0, 0],
        [0, 0, 0, np.exp(noisy_array_A[6] + 1j * noisy_theta_A[6] / 2) * cos(noisy_theta_A[6] / 2), 0, 0, 0, 0, 0, 0, 0, -sin(noisy_theta_A[6] / 2) * np.exp(1j * noisy_theta_A[6] / 2), 0, 0, 0, 0],
        [0, 0, 0, 0, np.exp(noisy_array_A[1] + 1j * noisy_theta_A[1] / 2) * cos(noisy_theta_A[1] / 2), 0, 0, 0, 0, 0, 0, 0, -sin(noisy_theta_A[1] / 2) * np.exp(1j * noisy_theta_A[1] / 2), 0, 0, 0],
        [0, 0, 0, 0, 0, np.exp(noisy_array_A[5] + 1j * noisy_theta_A[5] / 2) * cos(noisy_theta_A[5] / 2), 0, 0, 0, 0, 0, 0, 0, -sin(noisy_theta_A[5] / 2) * np.exp(1j * noisy_theta_A[5] / 2), 0, 0],
        [0, 0, 0, 0, 0, 0, np.exp(noisy_array_A[3] + 1j * noisy_theta_A[3] / 2) * cos(noisy_theta_A[3] / 2), 0, 0, 0, 0, 0, 0, 0, -sin(noisy_theta_A[3] / 2) * np.exp(1j * noisy_theta_A[3] / 2), 0],
        [0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_A[7] + 1j * noisy_theta_A[7] / 2) * cos(noisy_theta_A[7] / 2), 0, 0, 0, 0, 0, 0, 0, -sin(noisy_theta_A[7] / 2) * np.exp(1j * noisy_theta_A[7] / 2)]]

    noisy_B = np.array(1j) * [
        [np.exp(noisy_array_B[0] + 1j * noisy_theta_B[0] / 2) * sin(noisy_theta_B[0] / 2), 0, 0, 0, cos(noisy_theta_B[0] / 2) * np.exp(1j * noisy_theta_B[0] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, np.exp(noisy_array_B[2] + 1j * noisy_theta_B[2] / 2) * sin(noisy_theta_B[2] / 2), 0, 0, 0, cos(noisy_theta_B[2] / 2) * np.exp(1j * noisy_theta_B[2] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, np.exp(noisy_array_B[1] + 1j * noisy_theta_B[1] / 2) * sin(noisy_theta_B[1] / 2), 0, 0, 0, cos(noisy_theta_B[1] / 2) * np.exp(1j * noisy_theta_B[1] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, np.exp(noisy_array_B[3] + 1j * noisy_theta_B[3] / 2) * sin(noisy_theta_B[3] / 2), 0, 0, 0, cos(noisy_theta_B[3] / 2) * np.exp(1j * noisy_theta_B[3] / 2), 0, 0, 0, 0, 0, 0, 0, 0],
        [np.exp(noisy_array_B[0] + 1j * noisy_theta_B[0] / 2) * cos(noisy_theta_B[0] / 2), 0, 0, 0, -sin(noisy_theta_B[0] / 2) * np.exp(1j * noisy_theta_B[0] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, np.exp(noisy_array_B[2] + 1j * noisy_theta_B[2] / 2) * cos(noisy_theta_B[2] / 2), 0, 0, 0, -sin(noisy_theta_B[2] / 2) * np.exp(1j * noisy_theta_B[2] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, np.exp(noisy_array_B[1] + 1j * noisy_theta_B[1] / 2) * cos(noisy_theta_B[1] / 2), 0, 0, 0, -sin(noisy_theta_B[1] / 2) * np.exp(1j * noisy_theta_B[1] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, np.exp(noisy_array_B[3] + 1j * noisy_theta_B[3] / 2) * cos(noisy_theta_B[3] / 2), 0, 0, 0, -sin(noisy_theta_B[3] / 2) * np.exp(1j * noisy_theta_B[3] / 2), 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_B[4] + 1j * noisy_theta_B[4] / 2) * sin(noisy_theta_B[4] / 2), 0, 0, 0, cos(noisy_theta_B[4] / 2) * np.exp(1j * noisy_theta_B[4] / 2), 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_B[5] + 1j * noisy_theta_B[5] / 2) * sin(noisy_theta_B[5] / 2), 0, 0, 0, cos(noisy_theta_B[5] / 2) * np.exp(1j * noisy_theta_B[5] / 2), 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_B[6] + 1j * noisy_theta_B[6] / 2) * sin(noisy_theta_B[6] / 2), 0, 0, 0, cos(noisy_theta_B[6] / 2) * np.exp(1j * noisy_theta_B[6] / 2), 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_B[7] + 1j * noisy_theta_B[7] / 2) * sin(noisy_theta_B[7] / 2), 0, 0, 0, cos(noisy_theta_B[7] / 2) * np.exp(1j * noisy_theta_B[7] / 2)],
        [0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_B[4] + 1j * noisy_theta_B[4] / 2) * cos(noisy_theta_B[4] / 2), 0, 0, 0, -sin(noisy_theta_B[4] / 2) * np.exp(1j * noisy_theta_B[4] / 2), 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_B[5] + 1j * noisy_theta_B[5] / 2) * cos(noisy_theta_B[5] / 2), 0, 0, 0, -sin(noisy_theta_B[5] / 2) * np.exp(1j * noisy_theta_B[5] / 2), 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_B[6] + 1j * noisy_theta_B[6] / 2) * cos(noisy_theta_B[6] / 2), 0, 0, 0, -sin(noisy_theta_B[6] / 2) * np.exp(1j * noisy_theta_B[6] / 2), 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_B[7] + 1j * noisy_theta_B[7] / 2) * cos(noisy_theta_B[7] / 2), 0, 0, 0, -sin(noisy_theta_B[7] / 2) * np.exp(1j * noisy_theta_B[7] / 2)]]

    noisy_C = np.array(1j) * [
        [np.exp(noisy_array_C[0] + 1j * noisy_theta_C[0] / 2) * sin(noisy_theta_C[0] / 2), 0, cos(noisy_theta_C[0] / 2) * np.exp(1j * noisy_theta_C[0] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, np.exp(noisy_array_C[1] + 1j * noisy_theta_C[1] / 2) * sin(noisy_theta_C[1] / 2), 0, cos(noisy_theta_C[1] / 2) * np.exp(1j * noisy_theta_C[1] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [np.exp(noisy_array_C[0] + 1j * noisy_theta_C[0] / 2) * cos(noisy_theta_C[0] / 2), 0, -sin(noisy_theta_C[0] / 2) * np.exp(1j * noisy_theta_C[0] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, np.exp(noisy_array_C[1] + 1j * noisy_theta_C[1] / 2) * cos(noisy_theta_C[1] / 2), 0, -sin(noisy_theta_C[1] / 2) * np.exp(1j * noisy_theta_C[1] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, np.exp(noisy_array_C[2] + 1j * noisy_theta_C[2] / 2) * sin(noisy_theta_C[2] / 2), 0, cos(noisy_theta_C[2] / 2) * np.exp(1j * noisy_theta_C[2] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, np.exp(noisy_array_C[3] + 1j * noisy_theta_C[3] / 2) * sin(noisy_theta_C[3] / 2), 0, cos(noisy_theta_C[3] / 2) * np.exp(1j * noisy_theta_C[3] / 2), 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, np.exp(noisy_array_C[2] + 1j * noisy_theta_C[2] / 2) * cos(noisy_theta_C[2] / 2), 0, -sin(noisy_theta_C[2] / 2) * np.exp(1j * noisy_theta_C[2] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, np.exp(noisy_array_C[3] + 1j * noisy_theta_C[3] / 2) * cos(noisy_theta_C[3] / 2), 0, -sin(noisy_theta_C[3] / 2) * np.exp(1j * noisy_theta_C[3] / 2), 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_C[4] + 1j * noisy_theta_C[4] / 2) * sin(noisy_theta_C[4] / 2), 0, cos(noisy_theta_C[4] / 2) * np.exp(1j * noisy_theta_C[4] / 2), 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_C[5] + 1j * noisy_theta_C[5] / 2) * sin(noisy_theta_C[5] / 2), 0, cos(noisy_theta_C[5] / 2) * np.exp(1j * noisy_theta_C[5] / 2), 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_C[4] + 1j * noisy_theta_C[4] / 2) * cos(noisy_theta_C[4] / 2), 0, -sin(noisy_theta_C[4] / 2) * np.exp(1j * noisy_theta_C[4] / 2), 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_C[5] + 1j * noisy_theta_C[5] / 2) * cos(noisy_theta_C[5] / 2), 0, -sin(noisy_theta_C[5] / 2) * np.exp(1j * noisy_theta_C[5] / 2), 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_C[6] + 1j * noisy_theta_C[6] / 2) * sin(noisy_theta_C[6] / 2), 0, cos(noisy_theta_C[6] / 2) * np.exp(1j * noisy_theta_C[6] / 2), 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_C[7] + 1j * noisy_theta_C[7] / 2) * sin(noisy_theta_C[7] / 2), 0, cos(noisy_theta_C[7] / 2) * np.exp(1j * noisy_theta_C[7] / 2)],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_C[6] + 1j * noisy_theta_C[6] / 2) * cos(noisy_theta_C[6] / 2), 0, -sin(noisy_theta_C[6] / 2) * np.exp(1j * noisy_theta_C[6] / 2), 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_C[7] + 1j * noisy_theta_C[7] / 2) * cos(noisy_theta_C[7] / 2), 0, -sin(noisy_theta_C[7] / 2) * np.exp(1j * noisy_theta_C[7] / 2)]]

    noisy_D = np.array(1j) * [
        [np.exp(noisy_array_D[0] + 1j * noisy_theta_D[0] / 2) * sin(noisy_theta_D[0] / 2), cos(noisy_theta_D[0] / 2) * np.exp(1j * noisy_theta_D[0] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [np.exp(noisy_array_D[0] + 1j * noisy_theta_D[0] / 2) * cos(noisy_theta_D[0] / 2), -sin(noisy_theta_D[0] / 2) * np.exp(1j * noisy_theta_D[0] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, np.exp(noisy_array_D[1] + 1j * noisy_theta_D[1] / 2) * sin(noisy_theta_D[1] / 2), cos(noisy_theta_D[1] / 2) * np.exp(1j * noisy_theta_D[1] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, np.exp(noisy_array_D[1] + 1j * noisy_theta_D[1] / 2) * cos(noisy_theta_D[1] / 2), -sin(noisy_theta_D[1] / 2) * np.exp(1j * noisy_theta_D[1] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, np.exp(noisy_array_D[2] + 1j * noisy_theta_D[2] / 2) * sin(noisy_theta_D[2] / 2), cos(noisy_theta_D[2] / 2) * np.exp(1j * noisy_theta_D[2] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, np.exp(noisy_array_D[2] + 1j * noisy_theta_D[2] / 2) * cos(noisy_theta_D[2] / 2), -sin(noisy_theta_D[2] / 2) * np.exp(1j * noisy_theta_D[2] / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, np.exp(noisy_array_D[3] + 1j * noisy_theta_D[3] / 2) * sin(noisy_theta_D[3] / 2), cos(noisy_theta_D[3] / 2) * np.exp(1j * noisy_theta_D[3] / 2), 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, np.exp(noisy_array_D[3] + 1j * noisy_theta_D[3] / 2) * cos(noisy_theta_D[3] / 2), -sin(noisy_theta_D[3] / 2) * np.exp(1j * noisy_theta_D[3] / 2), 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_D[4] + 1j * noisy_theta_D[4] / 2) * sin(noisy_theta_D[4] / 2), cos(noisy_theta_D[4] / 2) * np.exp(1j * noisy_theta_D[4] / 2), 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_D[4] + 1j * noisy_theta_D[4] / 2) * cos(noisy_theta_D[4] / 2), -sin(noisy_theta_D[4] / 2) * np.exp(1j * noisy_theta_D[4] / 2), 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_D[5] + 1j * noisy_theta_D[5] / 2) * sin(noisy_theta_D[5] / 2), cos(noisy_theta_D[5] / 2) * np.exp(1j * noisy_theta_D[5] / 2), 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_D[5] + 1j * noisy_theta_D[5] / 2) * cos(noisy_theta_D[5] / 2), -sin(noisy_theta_D[5] / 2) * np.exp(1j * noisy_theta_D[5] / 2), 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_D[6] + 1j * noisy_theta_D[6] / 2) * sin(noisy_theta_D[6] / 2), cos(noisy_theta_D[6] / 2) * np.exp(1j * noisy_theta_D[6] / 2), 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_D[6] + 1j * noisy_theta_D[6] / 2) * cos(noisy_theta_D[6] / 2), -sin(noisy_theta_D[6] / 2) * np.exp(1j * noisy_theta_D[6] / 2), 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_D[7] + 1j * noisy_theta_D[7] / 2) * sin(noisy_theta_D[7] / 2), cos(noisy_theta_D[7] / 2) * np.exp(1j * noisy_theta_D[7] / 2)],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.exp(noisy_array_D[7] + 1j * noisy_theta_D[7] / 2) * cos(noisy_theta_D[7] / 2), -sin(noisy_theta_D[7] / 2) * np.exp(1j * noisy_theta_D[7] / 2)]]

    # noisy_U_fft = reduce(np.dot, [noisy_A, noisy_B, noisy_C, noisy_D])[0]
    # 获取加噪后的池化核序列，由于仅考虑了U模块噪声，因此需要除以4
    onn_kernel = np.abs(reduce(np.dot, [noisy_A, noisy_B, noisy_C, noisy_D])[0] / 4)

    # 用于测试池化无噪声
    # onn_kernel = np.abs(reduce(np.dot, [A, B, C, D])[0] / 4)
    # print(noisy_U_fft)
    # 0.001
    std1=[-0.25000655-1.08468777e-06j, -0.25000057-9.62185612e-07j,-0.24999845-1.15303994e-07j, -0.2499984 +1.73650632e-08j,-0.25000293+5.29491503e-07j, -0.25000084+6.55207964e-07j,-0.24999615+1.86747929e-06j, -0.24999874+1.99533962e-06j,-0.25000005+3.09610456e-06j, -0.2500032 +3.22797230e-06j,-0.24999997-3.86242177e-07j, -0.24999641-2.57779673e-07j,-0.24999844+3.13780576e-06j, -0.25000174+3.26071734e-06j,-0.25000377+2.11908800e-06j, -0.25000174+2.24895868e-06j]
    #0.01
    std2=[-0.25017215-3.78454292e-05j, -0.25010222-2.58169442e-05j,-0.24999648-3.21442380e-05j, -0.24998519-1.98116781e-05j,-0.25009437-3.24046299e-06j, -0.25006066+9.12975799e-06j, -0.25006962-8.75732237e-06j, -0.25001778+4.15320414e-06j,-0.25008441+2.66667198e-06j, -0.25002345+1.45353124e-05j,-0.24999694+4.64786765e-05j, -0.25000893+5.89023855e-05j,-0.25004437+5.32167451e-06j, -0.25001451+1.84345828e-05j,-0.25004022+2.40335680e-05j, -0.24998662+3.66183883e-05j]
    #0.1
    std3=[-0.25357136-2.80555649e-03j, -0.25178039-1.54758331e-03j,-0.25155191-1.36807535e-03j, -0.25043311-1.67896233e-04j,-0.25154929-1.53896418e-03j, -0.25013064-2.73205808e-04j,-0.24942054-1.59200294e-05j, -0.24886576+1.27494551e-03j,-0.25064427-1.40324479e-03j, -0.24957899-1.33269831e-04j,-0.24972204-1.41323936e-04j, -0.24859033+1.10522409e-03j,-0.25016406-7.38294304e-05j, -0.24861326+1.19018698e-03j,-0.24921013+8.05196962e-04j, -0.24707099+2.06232189e-03j]

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
                reset_phase_std=std_pooling,
                # device = torch.device("cpu"),
            )
            self.pool = nn.AvgPool2d(4, 4)
            self.linear = MZILinear(
                in_features=8 * 7 * 7,
                out_features=10,
                bias=True,
                # miniblock=4,
                mode="phase",
                decompose_alg="clements",
                photodetect=True,
                reset_phase_std=std_pooling,
                device=device,
            )
            self.bn = nn.BatchNorm2d(8, affine=True)

            self.conv.set_phase_variation(phase_noise)
            self.linear.set_phase_variation(phase_noise)
            # self.conv.reset_parameters()
            # self.linear.reset_parameters()

        def forward(self, x):
            # x = torch.relu(self.conv(x))
            x = torch.relu(self.bn(self.conv(x)))

            # 将上述的np数据类型转换为tensor以参与后续torch运算
            temp_kernel = torch.from_numpy(onn_kernel).float()
            # print(custom_kernel.shape)
            # 将池化核序列补充完整进行pytorch的跨步卷积运算来等效池化（8，1，4，4）在conv2d里面（通道，核个数，核尺寸1，核尺寸2）
            custom_kernel = temp_kernel.view(1, 1, 4, 4).expand(8, -1, -1, -1)
            # print(custom_kernel)
            custom_kernel = custom_kernel.float()
            Y = F.conv2d(x, custom_kernel, stride=4, padding=0, dilation=1, groups=8)
            x = Y.flatten(1)
            # x = self.linear(x)
            x = torch.relu(self.linear(x))
            return x

    # PART 3：创建一个CNN实例
    model = ONNModel()
    print("conv:", model.conv.phase_noise_std)
    print("Fc:", model.linear.phase_noise_std)
    # # 该函数包含了 SoftMax activation 和 cross entorpy，所以在神经网络结构定义的时候不需要定义softmax activation
    criterion = nn.CrossEntropyLoss()
    # # 第一个参数:我们想要训练的参数。
    # adam
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # SGD
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

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
            if (batch_idx + 1) % 10 == 0:
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
        # 循环遍历测试集中的图片值，images：图片像素，labels：图片上是数字几
        for images, labels in test_loader:
            outputs = model(images)  # model为ONN模型，将图片经过网络得到outputs
            _, predicted = torch.max(outputs.data, 1)  # outputs是十个值组成的向量，取值最大的为预测值
            total += labels.size(0)  # 总共输入到网络参与测试的图片数
            correct += (predicted == labels).sum().item()  # 若一张图片识别正确，correct+1
        # 用识别正确的图片数除以图片总数，得到准确率
        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
    print("FFT平均池化4×4", 'conv.mode= ', model.conv.mode, ' ,Linear.mode= ', model.linear.mode)
    # 在代码运行结束后调用 close 方法关闭日志文件
    console_logger.close()
    # 恢复原始的 sys.stdout
    sys.stdout = original_stdout
if __name__ == "__main__":
    input_OCNN = ONNModel()  # 类的实例化
    input_OCNN(data)  # 给网络输入图片
