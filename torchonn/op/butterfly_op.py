"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-11-27 19:23:38
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-11-27 19:25:28
"""
import math
from typing import Optional

import numpy as np
import torch
from pyutils.compute import complex_mult
from pyutils.general import logger

from torch import Tensor, nn
from torch.types import Device
import torch.fft

try:
    from torchonn.op.cuda_extension import universal_cuda
except:
    logger.warning(f"Import universal_cuda fail")
try:
    from torchonn.op.cuda_extension import hadamard_cuda
except:
    logger.warning(f"Import hadamard_cuda fail")

__all__ = [
    "hadamard_transform",
    "zero_bias_transform",
    "TrainableButterfly",
]


class FWHT1D_CPU(torch.autograd.Function):
    @staticmethod
    def transform(cls, tensor):
        """Simple implementation of FWHT, receiving as input a torch Tensor.
           快速沃尔什变换(FWHT) 
        """
        bit = length = len(tensor)
        result = tensor.detach().numpy()  # transform to numpy

        for _ in range(int(np.log2(length))):
            bit >>= 1
            for i in range(length):
                if i & bit == 0:
                    j = i | bit
                    temp = result[i]  # this copies by value
                    result[i] += result[j]
                    result[j] = temp - result[j]

        result /= np.sqrt(length)
        return torch.from_numpy(result)  # transform back to torch

    @staticmethod
    def forward(ctx, input):
        return FWHT1D_CPU.transform(input)

    @staticmethod
    def backward(ctx, grad_output):
        return FWHT1D_CPU.transform(grad_output)


class FWHT1D_CUDA(torch.autograd.Function):
    """Unitary 1D hadamard transform implemented with customized CUDA kernel. Normalization factor is 1/sqrt(N). N is a power of 2
    https://github.com/HazyResearch/structured-nets/blob/master/pytorch/structure/hadamard_cuda"""

    @staticmethod
    def forward(ctx, input):
        return hadamard_cuda.hadamard_transform(input) / np.sqrt(input.size(-1))

    @staticmethod
    def backward(ctx, grad_output):
        return hadamard_cuda.hadamard_transform(grad_output) / np.sqrt(grad_output.size(-1))


class UFT1D_CUDA(torch.autograd.Function):
    """Unitary 1D universal frequency transform implemented with customized CUDA kernel. Normalization factor is 1/sqrt(N). N is a power of 2"""

    @staticmethod
    def forward(ctx, input):
        factor = np.sqrt(input.size(-2))
        res = universal_cuda.universal_transform(input) / factor
        return res

    @staticmethod
    def backward(ctx, grad_output):
        factor = np.sqrt(grad_output.size(-2))
        res = universal_cuda.inverse_universal_transform(grad_output) / factor
        return res


class IUFT1D_CUDA(torch.autograd.Function):
    """Unitary 1D universal frequency transform implemented with customized CUDA kernel. Normalization factor is 1/sqrt(N). N is a power of 2"""

    @staticmethod
    def forward(ctx, input):
        factor = np.sqrt(input.size(-2))
        res = universal_cuda.inverse_universal_transform(input) / factor
        return res

    @staticmethod
    def backward(ctx, grad_output):
        factor = np.sqrt(grad_output.size(-2))
        res = universal_cuda.universal_transform(grad_output) / factor
        return res


def hadamard_transform(x: Tensor, complex: bool = False):
    """Hadamard butterfly transform 蝶形运算

    Args:
        x (cuda.Tensor or cuda.ComplexTensor): Complex tensors, real view of complex tensors, or real tensors.

    Returns:
        [cuda.Tensor or cuda.ComplexTensor]: Transformed tensor
    """
    if x.is_cuda():
        func = FWHT1D_CUDA.apply
    else:
        func = FWHT1D_CPU.apply
    if complex:
        if x.is_complex():  # complex tensor
            x = torch.complex(func(x.real), func(x.imag))
        else:  # real view of complex tensor
            x = torch.stack([func(x[..., 0]), func(x[..., 1])], dim=-1)
    else:  # real tensor
        x = func(x)

    return x


def zero_bias_transform(x: Tensor):
    """Zero bias butterfly transform

    Args:
        x (cuda.Tensor or cuda.ComplexTensor): Complex tensors or real view of complex tensors.

    Raises:
        NotImplementedError: Does not support CPU operators

    Returns:
        [cuda.Tensor or cuda.ComplexTensor]: Transformed tensor
    """
    if x.is_cuda():
        func = UFT1D_CUDA.apply
    else:
        raise NotImplementedError("UFT1D does not support device: CPU")
    if x.is_complex():
        x = torch.view_as_real(x)
        x = func(x)
        x = torch.view_as_complex(x)
    else:
        x = func(x)
    return x

# 可训练的蝶形网络结构
class TrainableButterfly(nn.Module):
    def __init__(
        self,
        length: int,
        reverse: bool = False,
        shared_phases: Optional[Tensor] = None,
        bit_reversal: bool = True,
        enable_last_level_phase_shifter: bool = True,
        coupler_transmission_factor_t: float = np.sqrt(2) / 2,
        coupler_insertion_loss: float = 0.0,
        crossing_transmission_factor: float = 1.0,
        crossing_phase_shift: float = 0.0,
        phase_quantizer: Optional[nn.Module] = None,
        device: Device = torch.device("cuda:0"),
    ):
        super(TrainableButterfly, self).__init__()
        self.length = length
        self.reverse = reverse
        self.n_level = int(np.log2(length))
        self.coupler_transmission_factor_t = coupler_transmission_factor_t
        self.coupler_insertion_loss = coupler_insertion_loss
        assert 0 <= self.coupler_insertion_loss <= 1, logger.error(
            f"Insertion loss of coupler should be within [0, 1], but got {self.coupler_insertion_loss}"
        )
        t = self.coupler_transmission_factor_t  # 耦合器的传输系数,默认t=np.sqrt(2) / 2
        insertion_loss = self.coupler_insertion_loss  # 耦合器的插入损耗 0
        k = np.sqrt(1 - insertion_loss - t ** 2)  # k=np.sqrt(2) / 2
        assert k >= 0, logger.error(
            f"Impossible transmission factor of coupler, requires t^2 + k^2 = 1 - insertion_loss, but got t={t}, insertion loss={insertion_loss}"
        )
        self.coupler_transmission_factor_k = k

        self.phase_quantizer = phase_quantizer

        self.bit_reversal = bit_reversal
        self.enable_last_level_phase_shifter = enable_last_level_phase_shifter
        self.crossing_transmission_factor = crossing_transmission_factor
        self.crossing_phase_shift = crossing_phase_shift

        self.device = device
        self.phases = (
            nn.Parameter(
                torch.zeros(
                    self.n_level + int(enable_last_level_phase_shifter),
                    length // 2,
                    2,
                    dtype=torch.float,
                    device=device,
                )
            )
            if shared_phases is None
            else shared_phases.data
        )
        self.permutations = ButterflyPermutation(
            length,
            crossing_transmission_factor=crossing_transmission_factor,
            crossing_phase_shift=crossing_phase_shift,
            device=device,
        )

        self.reset_parameters()
        self.eye = torch.eye(self.length, dtype=torch.cfloat, device=self.device).view(
            -1, self.length // 2, 2
        )
        self.dc_matrix = self.build_dc_matrix(t, k)  # t为实部,k为虚部

    def train_fft(self):
        inverse = self.reverse
        logger.info(f"Start initializing {'OFFT' if inverse == False else 'OIFFT'}")

        grad_state = self.phases.requires_grad
        perm_state = self.bit_reversal
        self.phases.requires_grad_(True)  # 设置self.phases.requires_grad_为True
        self.bit_reversal = True

        x = self.eye.flatten(1)  # 对输入矩阵一维化
        if inverse:  # 逆离散傅里叶变换
            target = torch.fft.ifft(x, n=self.length, dim=-1, norm="ortho")
            # print("IFFT")
        else:  # 离散傅里叶变换
            target = torch.fft.fft(x, n=self.length, dim=-1, norm="ortho")
            # print("FFT")

        if self.length == 4:
            self.phases.data.copy_(
                torch.tensor(
                    [
                        [0, -np.pi / 2, 0, -np.pi / 2],
                        [0, -np.pi / 2, -np.pi / 2, -3 * np.pi / 2],
                        [0, -np.pi / 2, 0, -np.pi / 2],
                    ]
                ).view(-1, self.length // 2, 2)
            )
            # print("self.phases", self.phases.shape)
        else:
            print("相位参数预训练！")
            optimizer = torch.optim.Adam((p for p in self.parameters() if p.requires_grad), lr=2e-3)
            x = self.eye.flatten(1)
            from tqdm import tqdm

            for step in tqdm(range(2000)):
                output = self.forward(x)
                # loss = torch.nn.functional.mse_loss(x, y)
                loss = output.sub(target).abs().square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            error = output.sub(target).norm(p=2).square() / target.norm(p=2).square()
            print(
                f"Finish initializing {'OFFT' if inverse == False else 'OIFFT'}, error = {error.data.item()}"
            )
        self.phases.requires_grad_(grad_state)
        self.bit_reversal = perm_state

    # 初始化该模块(OFFT\OIFFT)的相位矩阵
    def reset_parameters(self, alg: str = "uniform") -> Tensor:
        if alg == "uniform":
            nn.init.uniform_(self.phases, a=-np.pi / 2, b=-np.pi / 2)
        elif alg == "normal":
            nn.init.normal_(self.phases, 0, 1)
        elif alg == "zero":
            self.phases.data.fill_(0)
        elif alg == "hadamard":
            assert (
                self.enable_last_level_phase_shifter
            ), "Hadamard initialization is supported only when enable_last_level_phase_shifter is set to True"
            self.phases.data.fill_(0)
            self.phases.data[..., 1] = -np.pi / 2
        elif alg == "fft":
            assert (
                self.enable_last_level_phase_shifter
            ), "FFT initialization is supported only when enable_last_level_phase_shifter is set to True"
            self.train_fft()
        else:
            raise NotImplementedError

    def build_dc_matrix(self, t: float, k: float) -> Tensor:
        return torch.tensor([[t, k * 1j], [k * 1j, t]], device=self.device, dtype=torch.cfloat)

    # 返回该模块(OFFT\OIFFT)的传输矩阵
    def build_weight(self, phases: Optional[Tensor] = None) -> Tensor:
        phases = phases or self.phases  # [3,2,2]
        # print(phases.shape)
        if self.phase_quantizer is not None:
            phases = self.phase_quantizer(phases)

        weights = self.eye  # [4,2,2]
        # print(weights)
        if self.bit_reversal:
            shape = weights.size()
            # print(shape)
            weights = weights.flatten(1)  # [length, length]
            weights = self.permutations(weights, level=-1)
            weights = weights.reshape(shape)  # [length, length//2, 2]

        # print(self.phases[0])
        # print(self.phases[1])
        for level in range(self.n_level):  # 执行2次for循环，得到[4,4]的OFFT模块的矩阵
            # [length, length // 2, 2] * [length // 2, 2]
            weights = weights.mul(torch.exp(1j * self.phases[level]))  # 用初始权重（单位矩）乘以移相器矩阵
            # print("weights1", weights)
            weights = weights.matmul(
                self.dc_matrix  # 乘以分束器矩阵
            )
            # print("weights2", weights)
            # do not need to transpose dc matrix since it is symmetric 不需要转置dc矩阵，因为它是对称的
            # print(weights.shape)
            if level < self.n_level - 1:
                shape = weights.size()
                weights = weights.flatten(1)  # [length, length]
                weights = self.permutations(weights, level=level, inverse=self.reverse)
                weights = weights.reshape(shape)  # [length, length//2, 2]
        if self.enable_last_level_phase_shifter:
            weights = weights.mul(torch.exp(1j * self.phases[-1]))
        if self.bit_reversal:
            shape = weights.size()
            weights = weights.flatten(1)  # [length, length]
            weights = self.permutations(weights, level=self.n_level - 1)  # [length, length]
        else:
            weights = weights.flatten(1)  # [length, length]
        # print("weights", weights)
        # print(weights.shape) #[4,4]
        return weights  # [4,4]

    def forward(self, x, phases: Optional[Tensor] = None) -> Tensor:
        if not x.is_complex():
            x = torch.view_as_complex(x)  # 如果输入 x 不是复数形式，将其视为复数
        shape = x.size()  # [..., length]
        x = x.view(-1, self.length)  # [batch, length] 将输入 x 重塑为二维张量
        weights = self.build_weight(phases)  # 使用 build_weight 函数构建权重
        # print("weights.shape", weights.shape)
        x = x.matmul(weights.t())   # 对输入 x 进行矩阵乘法，得到输出 x
        x = x.view(shape)  # 将输出 x 重新调整为与输入相同的形状
        return x


class ButterflyPermutation(nn.Module):
    def __init__(
        self, length, crossing_transmission_factor=1, crossing_phase_shift=0, device=torch.device("cuda:0")
    ):
        super(ButterflyPermutation, self).__init__()
        self.length = length
        self.crossing_transmission_factor = crossing_transmission_factor
        assert 0 <= crossing_transmission_factor <= 1, logger.error(
            f"Transmission factor for waveguide crossings must be within [0, 1], but got {crossing_transmission_factor}"
        )
        self.crossing_phase_shift = crossing_phase_shift
        self.n_level = int(np.log2(self.length)) - 1
        if crossing_phase_shift < 1e-6 and crossing_transmission_factor > 1 - 1e-6:
            self.fast_forward = True
        else:
            self.fast_forward = False
        self.device = device

        self.forward_indices, self.backward_indices = self.gen_permutation_indices()
        self.bit_reversal_indices = bitreversal_permutation(self.length)
        self.num_crossings = self.calc_num_crossings(self.forward_indices)
        self.crossings = self.gen_crossings(self.num_crossings)

    def gen_permutation_indices(self):
        # forward indices  [1,2,3,4,5,6,7,8] -> [1,5,2,6,3,7,4,8]
        # barkward indices [1,2,3,4,5,6,7,8] -> [1,3,5,7,2,4,6,8]

        forward_indices, backward_indices = [], []
        initial_indices = torch.arange(0, self.length, dtype=torch.long, device=self.device)

        for level in range(self.n_level):
            block_size = 2 ** (level + 2)
            indices = (
                initial_indices.view(-1, self.length // block_size, 2, block_size // 2)
                .transpose(dim0=-2, dim1=-1)
                .contiguous()
                .view(-1)
            )
            forward_indices.append(indices)

            indices = initial_indices.view(-1, self.length // block_size, block_size)
            indices = torch.cat([indices[..., ::2], indices[..., 1::2]], dim=-1).contiguous().view(-1)
            backward_indices.append(indices)
        return forward_indices, backward_indices

    def calc_num_crossings(self, forward_indices):
        ### num crossings are related to forward indices
        ### for example
        ### from: 0 4 1 5 2 6 3 7
        ### to  : 0 1 2 3 4 5 6 7
        ### get : 0 3 1 2 2 1 3 0
        return [
            (indices - torch.arange(self.length, device=indices.device)).abs() for indices in forward_indices
        ]

    def gen_crossings(self, num_crossings):
        """
        @description: transfer matrix of cascaded crossings, modeling its insertion loss and phase shift
        @param num_crossings {list of torch.Tensor} number of crossings for all waveguides [length] * n_level
        @return: crossings {list of torch.Tensor} cascaded crossing transfer function [length, 2] * n_level
        """
        ### cascaded crossings (t^n)*(e^(n*phi))
        crossings = []
        for n_cross in num_crossings:
            n_cross = n_cross.float()
            mag = self.crossing_transmission_factor ** n_cross
            phase = n_cross * self.crossing_phase_shift
            # crossings.append(torch.stack([mag * phase.cos(), mag * phase.sin()], dim=-1))
            crossings.append(mag.mul(torch.exp(1j * phase)))
        return crossings

    def forward(self, x, level, inverse=False):
        if level == -1 or level == self.n_level:
            output = ButterflyPermutationFunction.apply(x, self.bit_reversal_indices)
        else:
            if inverse == False:
                # output = ButterflyPermutationFunction.apply(x, self.forward_indices[level], self.backward_indices[level])
                output = ButterflyPermutationFunction.apply(x, self.forward_indices[level])
                ## in the original transform, crossings are added after permutation
                # output = complex_mult(self.crossings[level][(None,) * (output.dim() - 2)], output)
                if not self.fast_forward:
                    output = self.crossings[level][(None,) * (output.dim() - 2)].mul(output)

            else:
                # output = ButterflyPermutationFunction.apply(x, self.backward_indices[self.n_level-level-1], self.forward_indices[self.n_level-level-1])
                ## in the reversed transform, crossings are added before permutation
                # x = complex_mult(self.crossings[level][(None,) * (x.dim() - 2)], x)
                if not self.fast_forward:
                    x = self.crossings[level][(None,) * (x.dim() - 2)].mul(x)
                output = ButterflyPermutationFunction.apply(
                    x, self.backward_indices[self.n_level - level - 1]
                )

        return output


def bitreversal_permutation(n: int, device: Device = torch.device("cuda:0")):
    """Return the bit reversal permutation used in FFT.
    Parameter:
        n: integer, must be a power of 2.
    Return:
        perm: bit reversal permutation, numpy array of size n
    """
    m = int(math.log2(n))
    assert n == 1 << m, "n must be a power of 2"
    perm = np.arange(n).reshape(n, 1)
    for i in range(m):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    return torch.from_numpy(perm.squeeze(0)).to(device)

#  该函数代码做出了部分调整，可以正确运行预训练
class ButterflyPermutationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, forward_indices):
        ctx.forward_indices = forward_indices.to(torch.long)
        output = input[..., ctx.forward_indices]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        forward_indices = ctx.forward_indices
        grad_input = grad_output.clone()
        grad_input[..., forward_indices] = grad_output
        return grad_input, None
