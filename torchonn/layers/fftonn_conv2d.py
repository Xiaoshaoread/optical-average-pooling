"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-11-28 00:13:10
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-11-28 00:23:47
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from pyutils.compute import im2col_2d, merge_chunks
from pyutils.general import logger, print_stat
from pyutils.quantize import input_quantize_fn, weight_quantize_fn
from torch import Tensor
from torch.nn import Parameter, init
from torch.types import Device, _size
from torch.nn.modules.utils import _pair

from torchonn.layers.base_layer import ONNBaseLayer
from torchonn.op.butterfly_op import TrainableButterfly
from torchonn.op.mzi_op import PhaseQuantizer

__all__ = [
    "FFTONNBlockConv2d",
]


class FFTONNBlockConv2d(ONNBaseLayer):
    """
    Butterfly blocking Conv2d layer.
    J. Gu, et al., "Towards Area-Efficient Optical Neural Networks: An FFT-based Architecture," ASP-DAC 2020.
    https://ieeexplore.ieee.org/document/9045156
    """

    _in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]
    miniblock: int
    __mode_list__ = ["fft", "hadamard", "zero_bias", "trainable"]

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size,
            stride: _size = 1,
            padding: _size = 0,
            dilation: _size = 1,
            groups: int = 1,
            bias: bool = True,
            miniblock: int = 4,  # 参数 miniblock 控制蝶式阵列中的小块大小
            mode: str = "fft",  # 参数 mode 控制卷积操作的模式
            photodetect: bool = True,
            device: Device = torch.device("cpu"),
    ):
        super().__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        assert groups == 1, f"Currently group convolution is not supported, but got group: {groups}"
        self.miniblock = miniblock
        self.in_channels_flat = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.grid_dim_x = int(np.ceil(self.in_channels_flat / miniblock))
        self.grid_dim_y = int(np.ceil(self.out_channels / miniblock))
        self.in_channels_pad = self.grid_dim_x * miniblock
        self.out_channels_pad = self.grid_dim_y * miniblock
        self.mode = mode
        assert mode in self.__mode_list__, logger.error(
            f"Mode not supported. Expected one from {self.__mode_list__} but got {mode}."
        )
        self.v_max = 10.8
        self.v_pi = 4.36
        self.gamma = np.pi / self.v_pi ** 2
        self.w_bit = 32
        self.in_bit = 32
        self.photodetect = photodetect

        crosstalk_filter_size = 3

        ### quantization tool
        self.input_quantizer = input_quantize_fn(self.in_bit, alg="dorefa", device=self.device)
        self.phase_U_quantizer = PhaseQuantizer(
            self.w_bit,
            self.v_pi,
            self.v_max,
            gamma_noise_std=0,
            crosstalk_factor=0,
            crosstalk_filter_size=crosstalk_filter_size,
            random_state=0,
            mode="butterfly",
            device=self.device,
        )
        self.phase_V_quantizer = PhaseQuantizer(
            self.w_bit,
            self.v_pi,
            self.v_max,
            gamma_noise_std=0,
            crosstalk_factor=0,
            crosstalk_filter_size=crosstalk_filter_size,
            random_state=0,
            mode="butterfly",
            device=self.device,
        )
        # self.phase_S_quantizer = PhaseQuantizer(
        #     self.w_bit,
        #     self.v_pi,
        #     self.v_max,
        #     gamma_noise_std=0,
        #     crosstalk_factor=0,
        #     crosstalk_filter_size=crosstalk_filter_size,
        #     random_state=0,
        #     mode="diagonal",
        #     device=self.device,
        # )

        ### build trainable parameters
        self.build_parameters()

        ### default set to slow forward
        self.disable_fast_forward()
        ### default set no phase variation
        self.set_phase_variation(0)
        ### default set no gamma noise
        self.set_gamma_noise(0)
        ### default set no crosstalk
        self.set_crosstalk_factor(0)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(mode=self.mode)

    def build_parameters(self) -> None:
        self.weight = torch.zeros(
            self.grid_dim_y,
            self.grid_dim_x,
            self.miniblock,
            self.miniblock,
            dtype=torch.cfloat,
            device=self.device,
        )

        # 创建TrainableButterfly对象，但不会执行reset_parameters()，因为没有指定形参alg
        self.T = TrainableButterfly(
            length=self.miniblock,  # 蝶式网络的长度，即处理的输入信号的维度
            reverse=False,  # 一个布尔值，表示是否进行逆蝶式操作。sxf
            bit_reversal=False,  # 一个布尔值，表示是否进行比特反转操作。
            enable_last_level_phase_shifter=True,  # 一个布尔值，表示是否启用最后一级的相位调整。
            phase_quantizer=self.phase_U_quantizer,
            device=self.device,
        )
        self.S = Parameter(
            torch.zeros(self.grid_dim_y, self.grid_dim_x, self.miniblock, dtype=torch.cfloat).to(self.device)
        )  # complex frequency-domain weights

        # 创建TrainableButterfly对象，但不会执行reset_parameters()，因为没有指定形参alg
        self.Tr = TrainableButterfly(
            length=self.miniblock,  # 蝶式网络的长度，即处理的输入信号的维度
            reverse=True,  # 一个布尔值，表示是否进行逆蝶式操作。sxf
            bit_reversal=False,  # 一个布尔值，表示是否进行比特反转操作。
            enable_last_level_phase_shifter=True,  # 一个布尔值，表示是否启用最后一级的相位调整。
            phase_quantizer=self.phase_V_quantizer,
            device=self.device,
        )

    @property
    def U(self):
        return self.Tr.build_weight()  # 返回OIFFT模块的传输矩阵

    @property
    def V(self):
        return self.T.build_weight()  # 返回OFFT模块的传输矩阵

    @property
    def phase_U(self):
        return self.Tr.phases  # 返回OIFFT模块的相位矩阵

    @property
    def phase_V(self):
        return self.T.phases  # 返回OFFT模块的相位矩阵

    def reset_parameters(self, mode: Optional[str] = None) -> None:
        mode = mode or self.mode
        W = init.kaiming_normal_(
            torch.empty(
                self.grid_dim_y,
                self.grid_dim_x,
                self.miniblock,
                self.miniblock,
                dtype=torch.cfloat,
                device=self.device,
            )
        )
        # print(W.shape)
        # print("卷积层权重", W)
        _, S, _ = torch.svd(W, compute_uv=False)

        self.S.data.copy_(S)  # 将奇异值分解的对角矩阵S作为权重的频域初始矩阵？
        print(S.shape)

        if mode == "zero_bias":
            self.T.reset_parameters(alg="zero")
            self.Tr.reset_parameters(alg="zero")
            self.T.phases.requires_grad_(False)
            self.Tr.phases.requires_grad_(False)
        elif mode == "hadamard":
            self.T.reset_parameters(alg="hadamard")
            self.Tr.reset_parameters(alg="hadamard")
            self.T.phases.requires_grad_(False)
            self.Tr.phases.requires_grad_(False)
        elif mode == "fft":
            # TrainableButterfly->reset_parameters()->train_fft() 即在初始化相位时就会执行预训练过程,如果length=4则指定相位
            self.T.reset_parameters(alg="fft")
            self.Tr.reset_parameters(alg="fft")
            self.T.phases.requires_grad_(False)  # self.T.reset_parameters(alg="fft")会设置phases.requires_grad_为True,这里重新进行关闭
            self.Tr.phases.requires_grad_(False)
        elif mode == "trainable":
            self.T.reset_parameters(alg="uniform")  # 初始化OFFT模块的相位矩阵
            self.Tr.reset_parameters(alg="uniform")  # 初始化IFFT模块的相位矩阵
            self.T.phases.requires_grad_(True)
            self.Tr.phases.requires_grad_(True)
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    def build_weight_from_usv(self, U: Tensor, S: Tensor, V: Tensor) -> Tensor:
        # differentiable feature is gauranteed
        # 用OFFT模块的传输矩阵乘以对角模块矩阵S,再乘以OIFFT模块的传输矩阵，得到分块的权重矩阵
        # print(U.shape)
        weight = U.matmul(S.unsqueeze(-1) * V)
        # print(weight.shape)
        self.weight.data.copy_(weight)  # ([2, 3, 4, 4])
        return weight

    def sync_parameters(self, src: str = "weight") -> None:
        """
        description: synchronize all parameters from the source parameters
        """
        self.weight.data.copy_(self.build_weight_from_usv(self.U.data, self.S.data, self.V.data))

    def build_weight(self, update_list: set = {"phase_U", "phase_S", "phase_V"}) -> Tensor:
        weight = self.build_weight_from_usv(self.U, self.S, self.V)

        return weight

    def set_gamma_noise(self, noise_std: float, random_state: Optional[int] = None) -> None:
        self.gamma_noise_std = noise_std
        self.phase_U_quantizer.set_gamma_noise(noise_std, self.phase_U.size(), random_state)
        # self.phase_S_quantizer.set_gamma_noise(noise_std, self.phase_S.size(), random_state)
        self.phase_V_quantizer.set_gamma_noise(noise_std, self.phase_V.size(), random_state)

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor
        self.phase_U_quantizer.set_crosstalk_factor(crosstalk_factor)
        # self.phase_S_quantizer.set_crosstalk_factor(crosstalk_factor)
        self.phase_V_quantizer.set_crosstalk_factor(crosstalk_factor)

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.phase_U_quantizer.set_bitwidth(w_bit)
        # self.phase_S_quantizer.set_bitwidth(w_bit)
        self.phase_V_quantizer.set_bitwidth(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        self.input_quantizer.set_bitwidth(in_bit)

    def load_parameters(self, param_dict: Dict[str, Any]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        super().load_parameters(param_dict=param_dict)

    def forward(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)   # 如果每个元素的位数小于16，则对输入 x 进行量化

        if not self.fast_forward_flag or self.weight is None:
            weight = self.build_weight()  # [p, q, k, k]
        else:
            weight = self.weight

        # print("weight1", weight)
        # 获取批量大小（bs），计算权重分割的偏移量
        bs = x.size(0)  # Batch_Size
        offset = int(np.ceil(self.grid_dim_x / 2)) * self.miniblock  # 8
        # 切片权重矩阵以获取相关部分
        weight = merge_chunks(weight)[
                 : self.out_channels, : self.in_channels * self.kernel_size[0] * self.kernel_size[1]
                 ]  # (8,9)
        # print("weight2", weight)
        # 对输入数据应用2D im2col转换
        _, x, h_out, w_out = im2col_2d(
            W=None,
            X=x,
            stride=self.stride[0],
            padding=self.padding[0],
            w_size=(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]),
        )
        # print(x.shape)  # (ker_size*ker_size,28*28*Batch_Size)
        x = x.to(torch.cfloat)  # 将输入数据 x 转换为复数
        # 将权重矩阵拆分
        weight_pos = weight[:, :offset]  # 包含了权重矩阵的8列（8，8）
        weight_neg = weight[:, offset:]  # 包含了权重矩阵的最后一列（8，1）
        # 将输入数据 x 拆分，并执行矩阵乘法
        x_pos = x[:offset, :]
        x_neg = x[offset:, :]
        x_pos = weight_pos.matmul(x_pos)
        x_pos = x_pos.real.square() + x_pos.imag.square()
        x_neg = weight_neg.matmul(x_neg)
        x_neg = x_neg.real.square() + x_neg.imag.square()
        x = x_pos - x_neg
        x = x.view(self.out_channels, h_out, w_out, bs).permute(3, 0, 1, 2).contiguous()

        if self.bias is not None:
                x = x + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return x
