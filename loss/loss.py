import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.transforms import Grayscale

from .msssim_loss import MSSSIM

_gradient_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

_gradient_y = _gradient_x.detach().clone().T

_gradient = (
    torch.stack([_gradient_x, _gradient_y])
    .reshape(2, 1, 3, 3)
    .to(torch.float32)
    # .to(torch.device("cuda"))
)


class Loss(nn.Module):

    def __init__(self, ssim_factor: float = 0.4, freq_factor: float = 0.1, grad_factor: float = 0.5,
                 device=None) -> None:
        super().__init__()
        self.ms_ssim = MSSSIM()
        self.ssim_factor = ssim_factor
        self.freq_factor = freq_factor
        self.grad_factor = grad_factor
        self.to_gray = Grayscale()
        self.grad_kernel = _gradient.to(device)

    def cal_grad(self, x: Tensor, factor: float = 0):
        grad: Tensor = self.to_gray(x)
        grad = F.conv2d(grad, self.grad_kernel, padding="same")
        return torch.sqrt(grad.pow(2).sum(dim=1, keepdim=True) + factor)

    def forward(self, x: Tensor, gt: Tensor, grad: Tensor) -> Tensor:
        l1 = F.l1_loss(x, gt)
        l2 = 1 - self.ms_ssim(x, gt)
        l3 = F.l1_loss(torch.fft.fft2(x), torch.fft.fft2(gt))
        l = l1 + self.ssim_factor * l2 + self.freq_factor * l3

        grad_gt = self.cal_grad(gt)
        grad_loss = F.l1_loss(grad, grad_gt)
        return l + grad_loss * self.grad_factor
