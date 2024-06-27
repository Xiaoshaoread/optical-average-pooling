"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 23:37:55
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 23:37:55
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from pyutils.compute import gen_gaussian_noise, merge_chunks
from pyutils.general import logger
from pyutils.quantize import input_quantize_fn
from torch import Tensor, asin, sin
from torch.nn import Parameter, init
from torch.types import Device, _size
from torch.nn.modules.utils import _pair
from torchonn.layers.base_layer import ONNBaseLayer
from torchonn.op.matrix_parametrization import RealUnitaryDecomposerBatch
from torchonn.op.mzi_op import (
    PhaseQuantizer,
    checkerboard_to_vector,
    phase_to_voltage,
    upper_triangle_to_vector,
    vector_to_checkerboard,
    vector_to_upper_triangle,
    voltage_to_phase,
)
import torch.autograd.profiler as profiler

__all__ = [
    "MZIConv2d",
]

# weights_MZI_conv2d = [] # <小邵>

class MZIConv2d(ONNBaseLayer):
    """
    SVD-based Conv2d layer constructed by cascaded MZIs.
    由级联mzi构建的基于svd的Conv2d层。
    """

    __constants__ = [
        # __constants__是一个函数属性，用于定义函数中的常量
        "stride",  # 横向步长
        "padding",  # 填充
        "dilation",  # 纵向步长
        "groups",  # 分组大小，默认为1
        "padding_mode",  # 填充模式
        "output_padding",  # 填充后的结果
        "in_channels",  # 输入通道数
        "out_channels",  # 输出通道数
        "kernel_size",  # 卷积核维数
        "reset_phase_std",
    ]

    __annotations__ = {"bias": Optional[torch.Tensor]}
    # 函数的bias参数可以是None或者一个torch.Tensor对象的实例。
    # __annotations__是一个特殊的字典属性，用于在函数定义中提供参数和返回值的类型注解。

    _in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    # tuple是一种有序、不可变的数据类型，通常用于存储一组相关的数据。tuple对象可以包含多个元素，每个元素可以是任何类型的Python对象，例如数字、字符串、列表、元组等。
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]
    reset_phase_std: int

    def __init__(
            # 对对象进行初始化操作
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size,
            stride: _size = 1,
            padding: _size = 0,
            dilation: _size = 1,
            groups: int = 1,
            bias: bool = True,
            mode: str = "weight",
            decompose_alg: str = "clements",
            photodetect: bool = True,
            reset_phase_std: int = 0,
            device: Device = torch.device("cpu"),
    ):
        super(MZIConv2d, self).__init__(device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)  # (kernel_size,kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.reset_phase_std = reset_phase_std
        self.groups = groups
        assert groups == 1, "Currently group convolution is not supported."
        self.mode = mode
        assert mode in {"weight", "usv", "phase", "voltage"}, logger.error(
            f"Mode not supported. Expected one from (weight, usv, phase, voltage) but got {mode}."
        )
        self.v_max = 10.8
        self.v_pi = 4.36
        self.gamma = np.pi / self.v_pi ** 2
        self.w_bit = 32
        self.in_bit = 32
        self.photodetect = photodetect  # 光电探测器
        self.decompose_alg = decompose_alg  # 矩阵分解模式

        # build trainable parameters 构建可训练的参数
        self.build_parameters(mode)

        # unitary parametrization tool 矩阵参数化工具
        # 《小邵》：RealUnitaryDecomposerBatch对象将被实例化为某种类型的分解器，用于分解给定的实数矩阵为实数酉矩阵的乘积的形式。
        self.decomposer = RealUnitaryDecomposerBatch(alg=decompose_alg)
        if decompose_alg == "clements":
            self.decomposer.v2m = vector_to_checkerboard
            self.decomposer.m2v = checkerboard_to_vector
            mesh_mode = "rectangle"  # 网络模型：矩形
            crosstalk_filter_size = 5
        elif decompose_alg in {"reck", "francis"}:
            self.decomposer.v2m = vector_to_upper_triangle
            self.decomposer.m2v = upper_triangle_to_vector
            mesh_mode = "triangle"  # 网络模型：三角
            crosstalk_filter_size = 3

        # quantization tool 量化
        self.input_quantizer = input_quantize_fn(self.in_bit, alg="dorefa", device=self.device)
        self.phase_U_quantizer = PhaseQuantizer(
            self.w_bit,
            self.v_pi,
            self.v_max,
            gamma_noise_std=0,
            crosstalk_factor=0,
            crosstalk_filter_size=crosstalk_filter_size,  # 滤波器
            random_state=0,
            mode=mesh_mode,
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
            mode=mesh_mode,
            device=self.device,
        )
        self.phase_S_quantizer = PhaseQuantizer(
            self.w_bit,
            self.v_pi,
            self.v_max,
            gamma_noise_std=0,
            crosstalk_factor=0,
            crosstalk_filter_size=crosstalk_filter_size,
            random_state=0,
            mode="diagonal",
            device=self.device,
        )

        # default set to slow forward 默认慢速前向传播
        self.disable_fast_forward()
        # default set no phase variation 默认没有相位变化
        self.set_phase_variation(0)  # 对相位设置高斯加性噪声phase_noise_std
        # default set no gamma noise 默认没有伽马噪声
        self.set_gamma_noise(0)  # 用于设置PhaseQuantizer量化器的量化噪声gamma_noise_std
        # default set no crosstalk 默认没有串扰
        self.set_crosstalk_factor(0)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    # 《小邵》：函数 build_parameters ：根据传入的 mode 参数创建相应的张量，并将其注册为模型参数参与训练和更新
    def build_parameters(self, mode: str = "weight") -> None:
        # weight mode
        weight = torch.Tensor(
            self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        ).to(self.device)
        # print(weight)
        # 这里构建权重矩阵
        # 后面在reset_parameters部分会进行初始化操作，

        # usv mode
        self.in_channels_flat = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        U = torch.Tensor(self.out_channels, self.out_channels).to(self.device)
        S = torch.Tensor(min(self.out_channels, self.in_channels_flat)).to(self.device)
        V = torch.Tensor(self.in_channels_flat, self.in_channels_flat).to(self.device)
        # phase mode 相位
        delta_list_U = torch.Tensor(self.out_channels).to(self.device)  # [8]
        phase_U = torch.Tensor(self.out_channels * (self.out_channels - 1) // 2).to(self.device)  # [28]
        phase_S = torch.Tensor(min(self.out_channels, self.in_channels_flat)).to(self.device)  # [8]
        delta_list_V = torch.Tensor(self.in_channels_flat).to(self.device)  # [9]
        phase_V = torch.Tensor(self.in_channels_flat * (self.in_channels_flat - 1) // 2).to(self.device)  # [36]
        # TIA gain 互阻放大器增益
        S_scale = torch.ones(1).to(self.device).float()  # [1]

        # 使用 Parameter 来包装模型参数，以便于在模型优化过程中对其进行自动求导、梯度更新等操作。<shao>
        if mode == "weight":
            self.weight = Parameter(weight)
        elif mode == "usv":
            # self.U = U
            # self.S = S
            # self.V = V
            self.U = Parameter(U)
            self.S = Parameter(S)
            self.V = Parameter(V)
        elif mode == "phase":
            #  构建相位参数，但不作为模型参数 <shao>
            self.phase_U = phase_U
            self.phase_S = phase_S
            self.phase_V = phase_V
            self.S_scale = S_scale
            #  构建模型参数，用于模型参数的直接更新 <shao>
            self.U = Parameter(U)
            self.S = Parameter(S)
            self.V = Parameter(V)

        elif mode == "voltage":  # 电压
            raise NotImplementedError
        else:
            raise NotImplementedError

        for p_name, p in {
            "weight": weight,
            "U": U,
            "S": S,
            "V": V,
            "phase_U": phase_U,
            "phase_S": phase_S,
            "phase_V": phase_V,
            "S_scale": S_scale,
            "delta_list_U": delta_list_U,
            "delta_list_V": delta_list_V,
        }.items():
            if not hasattr(self, p_name):
                self.register_buffer(p_name, p)

    # 权重参数的初始化
    def reset_parameters(self) -> None:  # 参数初始化
        # profiler = torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #     record_shapes=True
        # )
        if self.mode == "weight":
            a = init.kaiming_normal_(self.weight.data)  # 权重初始化
            # print(a)
        elif self.mode == "usv":
            # 设定随机种子
            torch.manual_seed(0)
            W = init.kaiming_normal_(
                torch.empty(self.out_channels, self.in_channels_flat, dtype=self.U.dtype, device=self.device)
            )
            # std_dev = self.reset_phase_std
            # print("conv_初始化std :", std_dev)
            # with torch.no_grad():
            #     W += torch.normal(mean=0, std=std_dev, size=W.size())

            # W = init.xavier_normal_(
            #     torch.empty(self.out_channels, self.in_channels_flat, dtype=self.U.dtype, device=self.device)
            # )
            # W = torch.zeros(self.out_channels, self.in_channels_flat, dtype=self.U.dtype, device=self.device)
            # print("conv初始化卷积核",W)
            # self.weight.data.copy_(W)
            U, S, V = torch.svd(W, some=False)
            V = V.transpose(-2, -1)
            std_dev = self.reset_phase_std
            print("conv_std初始化制造误差 :", std_dev)
            decomposer = RealUnitaryDecomposerBatch(alg="clements")
            # U, S, V = torch.linalg.svd(W, full_matrices=True)  # must use QR decomposition
            delta_list, phi_mat = self.decomposer.decompose(U)
            self.delta_list_U.data.copy_(delta_list)
            self.phase_U.data.copy_(self.decomposer.m2v(phi_mat))
            self.phase_U = self.phase_U + torch.randn(self.phase_U.size(), dtype=self.phase_U.dtype,
                                                      device=self.phase_U.device) * std_dev

            delta_list, phi_mat = self.decomposer.decompose(V)
            self.delta_list_V.data.copy_(delta_list)
            self.phase_V.data.copy_(self.decomposer.m2v(phi_mat))
            self.phase_V = self.phase_V + torch.randn(self.phase_V.size(), dtype=self.phase_V.dtype,
                                                      device=self.phase_V.device) * std_dev

            self.S_scale.data.copy_(S.abs().max(dim=-1, keepdim=True)[0])
            self.phase_S.data.copy_(S.div(self.S_scale.data).acos())

            ###
            mat_phase_U = vector_to_checkerboard(self.phase_U)
            U = decomposer.reconstruct(self.delta_list_U, mat_phase_U)
            mat_phase_V = vector_to_checkerboard(self.phase_V)
            V = decomposer.reconstruct(self.delta_list_V, mat_phase_V)
            self.U.data.copy_(U)
            self.V.data.copy_(V)
            self.S.data.copy_(torch.ones(S.shape[0], dtype=self.U.dtype, device=self.device))

        elif self.mode == "phase":
            torch.manual_seed(1234)  # 设置随机种子
            W = init.kaiming_normal_(
                torch.empty(self.out_channels, self.in_channels_flat, dtype=self.U.dtype, device=self.device)
            )  # [8,9]

            # W = init.xavier_uniform_(
            #     torch.empty(self.out_channels, self.in_channels_flat, dtype=self.U.dtype, device=self.device)
            # )
            # print("linear初始化权重矩阵", W)
            U, S, V = torch.svd(W, some=False)  # U[8, 8],S[8],V[9, 9]
            V = V.transpose(-2, -1)  # V[9, 9] 转置

            std_dev = self.reset_phase_std
            # U, S, V = torch.linalg.svd(W, full_matrices=True)  # must use QR decomposition
            delta_list, phi_mat = self.decomposer.decompose(U)
            self.delta_list_U.data.copy_(delta_list)
            phase_U = self.decomposer.m2v(phi_mat)
            self.phase_U = phase_U + torch.randn(phase_U.size(), dtype=phase_U.dtype, device=phase_U.device) * std_dev

            delta_list, phi_mat = self.decomposer.decompose(V)
            self.delta_list_V.data.copy_(delta_list)
            phase_V = self.decomposer.m2v(phi_mat)
            self.phase_V = phase_V + torch.randn(phase_V.size(), dtype=phase_V.dtype, device=phase_V.device) * std_dev

            self.S_scale.data.copy_(S.abs().max(dim=-1, keepdim=True)[0])
            self.phase_S.data.copy_(S.div(self.S_scale.data).acos())

            # 合成USV矩阵
            decomposer = RealUnitaryDecomposerBatch(alg="clements")
            mat_phase_U = vector_to_checkerboard(self.phase_U)
            U = decomposer.reconstruct(self.delta_list_U, mat_phase_U)
            mat_phase_V = vector_to_checkerboard(self.phase_V)
            V = decomposer.reconstruct(self.delta_list_V, mat_phase_V)
            #  根据初始化的权重weight获取模型参数self.U、self.S、self.V,用于后面模型的训练于更新
            self.U.data.copy_(U)
            self.V.data.copy_(V)
            self.S.data.copy_(torch.ones(S.shape[0], dtype=self.U.dtype, device=self.device))


        elif self.mode == "voltage":
            raise NotImplementedError
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    # 《小邵》：根据SVD的结果U、S、V来构建权重张量weight的
    def build_weight_from_usv(self, U: Tensor, S: Tensor, V: Tensor) -> Tensor:
        #  differentiable feature is gauranteed
        if self.out_channels == self.in_channels_flat:  # 《小邵》：in_channels_flat理解为MZI网格结构的输入端口数
            weight = torch.mm(U, S.unsqueeze(1) * V)
        elif self.out_channels > self.in_channels_flat:
            weight = torch.mm(U[:, : self.in_channels_flat], S.unsqueeze(1) * V)
        else:
            # 运行
            # print(U)
            # print(S.unsqueeze(0))
            weight = torch.mm(U * S.unsqueeze(0), V[: self.out_channels, :])
        # self.weight.data.copy_(weight)

        # 《小邵》：在这个函数中，如果MZI的输出端口数和输入端口数相等，直接用SVD分解的结果计算weight；
        # 如果输出通道数比输入通道数小，则只使用前output_channel个奇异值；
        # 如果输出通道数比输入通道数大，则使用所有的奇异值。最后返回weight。
        # 结合MZI网格结构来理解
        return weight

    #  《小邵》：根据MZI的具体相位值来构建权重张量weight
    def build_weight_from_phase(
            self,
            delta_list_U: Tensor,
            phase_U: Tensor,
            delta_list_V: Tensor,
            phase_V: Tensor,
            phase_S: Tensor,
            update_list: set = {"phase_U", "phase_S", "phase_V"},
    ) -> Tensor:
        ### not differentiable   不可微分
        ### reconstruct is time-consuming, a fast method is to only reconstruct based on updated phases
        # 重构耗时长，一种快速的方法是只基于已更新的相位进行重构
        if "phase_U" in update_list:
            self.U.data.copy_(self.decomposer.reconstruct(delta_list_U, self.decomposer.v2m(phase_U)))
            # print("update_self.U", self.U)
        if "phase_V" in update_list:
            self.V.data.copy_(self.decomposer.reconstruct(delta_list_V, self.decomposer.v2m(phase_V)))
        if "phase_S" in update_list:
            self.S.data.copy_(phase_S.cos().mul_(self.S_scale))
            # print("update_self.S", self.S)
        return self.build_weight_from_usv(self.U, self.S, self.V)

    #  《小邵》：此函数不做考虑
    def build_weight_from_voltage(
            self,
            delta_list_U: Tensor,
            voltage_U: Tensor,
            delta_list_V: Tensor,
            voltage_V: Tensor,
            voltage_S: Tensor,
            gamma_U: Union[float, Tensor],
            gamma_V: Union[float, Tensor],
            gamma_S: Union[float, Tensor],
    ) -> Tensor:
        self.phase_U = voltage_to_phase(voltage_U, gamma_U)
        self.phase_V = voltage_to_phase(voltage_V, gamma_V)
        self.phase_S = voltage_to_phase(voltage_S, gamma_S)
        return self.build_weight_from_phase(
            delta_list_U, self.phase_U, delta_list_V, self.phase_V, self.phase_S
        )

    #  《小邵》：根据输入U、S、V来获取其相位 phi_mat 和对角元素信息 delta_list
    def build_phase_from_usv(
            self, U: Tensor, S: Tensor, V: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        #  《小邵》：使用 decompose 将U矩阵分解为相位和对角元素信息
        delta_list, phi_mat = self.decomposer.decompose(U.data.clone())
        self.delta_list_U.data.copy_(delta_list)
        self.phase_U.data.copy_(self.decomposer.m2v(phi_mat))

        #  《小邵》：使用 decompose 将V矩阵分解为相位和增幅信息
        delta_list, phi_mat = self.decomposer.decompose(V.data.clone())
        self.delta_list_V.data.copy_(delta_list)
        self.phase_V.data.copy_(self.decomposer.m2v(phi_mat))

        self.S_scale.data.copy_(S.data.abs().max(dim=-1, keepdim=True)[0])
        self.phase_S.data.copy_(S.data.div(self.S_scale.data).acos())

        return self.delta_list_U, self.phase_U, self.delta_list_V, self.phase_V, self.phase_S, self.S_scale

    def build_usv_from_phase(
            self,
            delta_list_U: Tensor,
            phase_U: Tensor,
            delta_list_V: Tensor,
            phase_V: Tensor,
            phase_S: Tensor,
            S_scale: Tensor,
            update_list: Dict = {"phase_U", "phase_S", "phase_V"},
    ) -> Tuple[Tensor, ...]:
        ### not differentiable
        # reconstruct is time-consuming, a fast method is to only reconstruct based on updated phases
        if "phase_U" in update_list:
            self.U.data.copy_(self.decomposer.reconstruct(delta_list_U, self.decomposer.v2m(phase_U)))
        if "phase_V" in update_list:
            self.V.data.copy_(self.decomposer.reconstruct(delta_list_V, self.decomposer.v2m(phase_V)))
        if "phase_S" in update_list:
            self.S.data.copy_(phase_S.data.cos().mul_(S_scale))
        return self.U, self.S, self.V

    def build_usv_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        ### differentiable feature is gauranteed
        U, S, V = weight.data.svd(some=False)
        V = V.transpose(-2, -1).contiguous()
        self.U.data.copy_(U)
        self.S.data.copy_(S)
        self.V.data.copy_(V)
        return U, S, V

    def build_phase_from_weight(
            self, weight: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.build_phase_from_usv(*self.build_usv_from_weight(weight))

    def build_voltage_from_phase(
            self,
            delta_list_U: Tensor,
            phase_U: Tensor,
            delta_list_V: Tensor,
            phase_V: Tensor,
            phase_S: Tensor,
            S_scale: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        self.delta_list_U = delta_list_U
        self.delta_list_V = delta_list_V
        self.voltage_U.data.copy_(phase_to_voltage(phase_U, self.gamma))
        self.voltage_S.data.copy_(phase_to_voltage(phase_S, self.gamma))
        self.voltage_V.data.copy_(phase_to_voltage(phase_V, self.gamma))
        self.S_scale.data.copy_(S_scale)

        return (
            self.delta_list_U,
            self.voltage_U,
            self.delta_list_V,
            self.voltage_V,
            self.voltage_S,
            self.S_scale,
        )

    def build_voltage_from_usv(
            self, U: Tensor, S: Tensor, V: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.build_voltage_from_phase(*self.build_phase_from_usv(U, S, V))

    def build_voltage_from_weight(
            self, weight: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.build_voltage_from_phase(*self.build_phase_from_usv(*self.build_usv_from_weight(weight)))

    def sync_parameters(self, src: str = "weight") -> None:
        """
        description: synchronize all parameters from the source parameters
        同步源参数中的所有参数
        """
        print("sync!已执行")
        if src == "weight":
            self.build_phase_from_weight(self.weight)
        elif src == "usv":
            self.build_phase_from_usv(self.U, self.S, self.V)
            self.build_weight_from_usv(self.U, self.S, self.V)
        elif src == "phase":
            if self.w_bit < 16:
                phase_U = self.phase_U_quantizer(self.phase_U.data)
                phase_S = self.phase_S_quantizer(self.phase_S.data)
                phase_V = self.phase_V_quantizer(self.phase_V.data)
            else:
                phase_U = self.phase_U
                phase_S = self.phase_S
                phase_V = self.phase_V
            if self.phase_noise_std > 1e-5:
                ### phase_S is assumed to be protected
                phase_U = phase_U + gen_gaussian_noise(
                    phase_U,
                    0,
                    self.phase_noise_std,
                    trunc_range=(-2 * self.phase_noise_std, 2 * self.phase_noise_std),
                )
                phase_V = phase_V + gen_gaussian_noise(
                    phase_V,
                    0,
                    self.phase_noise_std,
                    trunc_range=(-2 * self.phase_noise_std, 2 * self.phase_noise_std),
                )

            self.build_weight_from_phase(
                self.delta_list_U, phase_U, self.delta_list_V, phase_V, phase_S, self.S_scale
            )
        elif src == "voltage":
            NotImplementedError
        else:
            raise NotImplementedError

    #  每进行一次参数更新后都会执行该正向传播
    def build_weight(self, update_list: set = {"phase_U", "phase_S", "phase_V"}) -> Tensor:
        # with profiler.profile(record_shapes=True, use_cuda=False) as prof:
        if self.mode == "weight":
            weight = self.weight
        elif self.mode == "usv":
            U = self.U
            V = self.V
            S = self.S
            weight = self.build_weight_from_usv(U, S, V)
            # print("weight", self.weight)
            # print("S",S)
        elif self.mode == "phase":
            # print("self.U", self.U)
            #  从已更新的模型参数self.U、self.S、self.V来间接的获取更新后的相位，并存储在内部参数self.phase_U、self.phase_S、self.phase_V
            delta_list, phi_mat = self.decomposer.decompose(self.U.detach().numpy())
            self.delta_list_U.data.copy_(torch.from_numpy(delta_list))
            self.phase_U.data.copy_(self.decomposer.m2v(torch.from_numpy(phi_mat)))

            # print("delta_list_U", self.delta_list_U)
            # print("self.phase_U", self.phase_U)

            delta_list, phi_mat = self.decomposer.decompose(self.V.detach().numpy())
            self.delta_list_V.data.copy_(torch.from_numpy(delta_list))
            self.phase_V.data.copy_(self.decomposer.m2v(torch.from_numpy(phi_mat)))

            self.S_scale.data.copy_(self.S.abs().max(dim=-1, keepdim=True)[0])
            self.phase_S.data.copy_(self.S.div(self.S_scale.data).acos())
            # print("S", self.S)
            # print("self.phase_S", self.phase_S)
            # print("S_scale", self.S_scale)

            # print(self.phase_U)

            ### not differentiable
            if self.w_bit < 16 or self.gamma_noise_std > 1e-5 or self.crosstalk_factor > 1e-5:
                phase_U = self.phase_U_quantizer(self.phase_U.data)
                # print("phase_U_quantizer", phase_U)
                phase_S = self.phase_S_quantizer(self.phase_S.data)
                phase_V = self.phase_V_quantizer(self.phase_V.data)

            else:
                phase_U = self.phase_U
                phase_S = self.phase_S
                phase_V = self.phase_V

            #  tips:这里上下两种噪声不可同时添加，因为下面的phase_u会覆盖上面的phase_u，而不是叠加。

            #  对（间接）更新后的相位值引入高斯噪声
            #  phase_U = self.phase_U + gen_gaussian_noise这里表明，噪声是作用在更新后的相位上，即左边为phase右边为self.phase
            if self.phase_noise_std > 1e-5:
                ### phase_S is assumed to be protected
                phase_U = self.phase_U + gen_gaussian_noise(
                    phase_U,  # 待加入噪声的相位
                    0,  # 噪声均值
                    self.phase_noise_std,  # 噪声标准差
                    trunc_range=(-2 * self.phase_noise_std, 2 * self.phase_noise_std),
                )
                phase_V = self.phase_V + gen_gaussian_noise(
                    phase_V,
                    0,
                    self.phase_noise_std,
                    trunc_range=(-2 * self.phase_noise_std, 2 * self.phase_noise_std),
                )
                # print("对卷积层噪声")

            # print("self.weight", self.weight)
            weight = self.build_weight_from_phase(
                self.delta_list_U, phase_U, self.delta_list_V, phase_V, self.phase_S, update_list=update_list
            )
            # print(weight.shape)
            # self.weight = weight  # 方便调用 <shao>
            # print("build_weight_from_phase_weight", weight)
            # u, s, v = torch.svd(weight, some=False)
            # print("u", u)
            # self.U = torch.nn.Parameter(torch.svd(weight, some=False)[0])
            # self.S = torch.nn.Parameter(torch.svd(weight, some=False)[1])
            # self.V = torch.nn.Parameter(torch.svd(weight, some=False)[2])


        elif self.mode == "voltage":
            raise NotImplementedError
        else:
            raise NotImplementedError
        # print("built_weight:" + str(weight))
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        return weight

    #  《小邵》：函数首先将 self.gamma_noise_std 设置为输入的噪声标准差。
    #  然后分别使用输入的标准差和随机数种子设置了三个量化器（phase_U_quantizer、phase_S_quantizer 和 phase_V_quantizer）的 gamma 噪声
    def set_gamma_noise(self, noise_std: float, random_state: Optional[int] = None) -> None:
        self.gamma_noise_std = noise_std
        self.phase_U_quantizer.set_gamma_noise(noise_std, self.phase_U.size(), random_state)
        self.phase_S_quantizer.set_gamma_noise(noise_std, self.phase_S.size(), random_state)
        self.phase_V_quantizer.set_gamma_noise(noise_std, self.phase_V.size(), random_state)

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor
        self.phase_U_quantizer.set_crosstalk_factor(crosstalk_factor)
        self.phase_S_quantizer.set_crosstalk_factor(crosstalk_factor)
        self.phase_V_quantizer.set_crosstalk_factor(crosstalk_factor)

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.phase_U_quantizer.set_bitwidth(w_bit)
        self.phase_S_quantizer.set_bitwidth(w_bit)
        self.phase_V_quantizer.set_bitwidth(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        self.input_quantizer.set_bitwidth(in_bit)

    def load_parameters(self, param_dict: Dict[str, Any]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        print("load_parameters")
        super().load_parameters(param_dict=param_dict)
        if self.mode == "phase":
            self.build_weight(update_list=param_dict)

    def get_output_dim(self, img_height: int, img_width: int) -> _size:
        # 得到输出的行数、列数
        h_out = (
                        img_height - self.dilation[0] * (self.kernel_size[0] - 1) - 1 + 2 * self.padding[0]
                ) / self.stride[0] + 1
        w_out = (
                        img_width - self.dilation[1] * (self.kernel_size[1] - 1) - 1 + 2 * self.padding[1]
                ) / self.stride[1] + 1
        return int(h_out), int(w_out)

    def forward(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)
        if not self.fast_forward_flag or self.weight is None:
            # 运行
            weight = self.build_weight()  # [out_channels, in_channels_flat]
        else:
            weight = self.weight
        # print("conv_weight: "  + str(weight))

        weight = weight.view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        # print("forward",weight)
        self.weight = weight
        x = F.conv2d(
            x,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.photodetect:  # 如果有光电探测器
            x = x.square()

        if self.bias is not None:
            x = x + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # print("weight: " + str(weight))
        # print("self.weight: " + str(self.weight))
        return x
