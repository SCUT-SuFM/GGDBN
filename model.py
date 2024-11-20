from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torchvision.transforms import Grayscale


def weights_init(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """

    def init_conv(m):
        if init_type == "normal":
            torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == "xavier":
            torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == "kaiming":
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif init_type == "orthogonal":
            torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError(
                "initialization method [%s] is not implemented" % init_type
            )
        if m.bias is not None and m.bias.data is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init_conv(m)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print("initialize network with %s type" % init_type)


class Conv2dLayer(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            bias: bool = True,
            padding: _size_2_t | str = 0,
            padding_mode: str = "zeros",
            device=None,
            dtype=None,
            norm_op: Optional[Callable[..., nn.Module]] = None,
            act_op: Optional[Callable[..., nn.Module]] = nn.PReLU,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              bias=bias, padding=padding, padding_mode=padding_mode, device=device, dtype=dtype)
        if norm_op is None:
            self.norm = None
        else:
            self.norm = norm_op(out_channels)

        if act_op is None:
            self.act = None
        else:
            self.act = act_op()

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class DenseResidualBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            padding: _size_2_t | str = 0,
            bias: bool = True,
            padding_mode: str = "zeros",
            device=None,
            dtype=None,
            norm_op: Optional[Callable[..., nn.Module]] = None,
            act_op: Optional[Callable[..., nn.Module]] = nn.PReLU,
    ):
        super().__init__()
        self.conv1 = Conv2dLayer(in_channels, int(in_channels / 2.0), kernel_size,
                                 padding=padding, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype,
                                 norm_op=norm_op, act_op=act_op)
        self.conv2 = Conv2dLayer(int(in_channels * 3 / 2.0), int(in_channels / 2.0), kernel_size,
                                 padding=padding, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype,
                                 norm_op=norm_op, act_op=act_op)
        self.conv3 = Conv2dLayer(in_channels * 2, out_channels, kernel_size,
                                 padding=padding, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype,
                                 norm_op=norm_op, act_op=act_op)

    def forward(self, x):
        idt = x
        out = self.conv1(x)
        cat = torch.cat((x, out), dim=1)
        out = self.conv2(cat)
        cat = torch.cat((cat, out), dim=1)
        out = self.conv3(cat)
        return out + idt


class SGNBase(nn.Module):

    def __init__(
            self,
            in_channels: int,
            inter_channels: int,
            padding_mode: str,
            norm_op: Optional[Callable[..., nn.Module]] = None,
            act_op: Optional[Callable[..., nn.Module]] = nn.PReLU,
    ):
        super().__init__()
        # Top network, K = 3
        self.top1 = Conv2dLayer(in_channels * (4 ** 3), inter_channels * (2 ** 3), 3,
                                padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)
        self.top2 = DenseResidualBlock(inter_channels * (2 ** 3), inter_channels * (2 ** 3), 3,
                                       padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)
        self.top3 = Conv2dLayer(inter_channels * (2 ** 3), inter_channels * (2 ** 3), 3,
                                padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)
        # Middle subnetwork, K = 2
        self.mid1 = Conv2dLayer(in_channels * (4 ** 2), inter_channels * (2 ** 2), 3,
                                padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)
        self.mid2 = Conv2dLayer(int(inter_channels * (2 ** 2 + 2 ** 3 / 4.0)), inter_channels * (2 ** 2), 3,
                                padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)
        self.mid3 = DenseResidualBlock(inter_channels * (2 ** 2), inter_channels * (2 ** 2), 3,
                                       padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)
        self.mid4 = Conv2dLayer(inter_channels * (2 ** 2), inter_channels * (2 ** 2), 3,
                                padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)
        # Bottom subnetwork, K = 1
        self.bot1 = Conv2dLayer(in_channels * (4 ** 1), inter_channels * (2 ** 1), 3,
                                padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)
        self.bot2 = Conv2dLayer(int(inter_channels * (2 ** 1 + 2 ** 2 / 4.0)), inter_channels * (2 ** 1), 3,
                                padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)

        self.bot3 = DenseResidualBlock(inter_channels * (2 ** 1), inter_channels * (2 ** 1), 3,
                                       padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)
        self.bot4 = Conv2dLayer(inter_channels * (2 ** 1), inter_channels * (2 ** 1), 3,
                                padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)

    def forward(self, x):
        # x: B * 3 * 256 * 256
        # x1: B * 12 * 128 * 128
        x1 = F.pixel_unshuffle(x, 2)
        # x2: B * 48 * 64 * 64
        x2 = F.pixel_unshuffle(x, 4)
        # x3: B * 192 * 32 * 32
        x3 = F.pixel_unshuffle(x, 8)
        # Top
        x3 = self.top1(x3)
        x3 = self.top2(x3)
        x3 = self.top3(x3)
        x3 = F.pixel_shuffle(x3, 2)
        # Middle
        x2 = self.mid1(x2)
        x2 = torch.cat((x2, x3), dim=1)
        x2 = self.mid2(x2)
        x2 = self.mid3(x2)
        x2 = self.mid4(x2)
        x2 = F.pixel_shuffle(x2, 2)
        # bottom
        x1 = self.bot1(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.bot2(x1)
        x1 = self.bot3(x1)
        x1 = self.bot4(x1)
        x1 = F.pixel_shuffle(x1, 2)
        return x1


class Attention(nn.Module):

    def __init__(self, channels, reduction=16, spatial_kernel=7, mode="channel"):
        super().__init__()
        assert mode in ["channel", "spatial", "cbam"]
        self.mode = mode
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.PReLU(),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

        # spatial attention
        if mode == "spatial" or mode == "cbam":
            self.conv = nn.Conv2d(
                2,
                1,
                kernel_size=spatial_kernel,
                padding=spatial_kernel // 2,
                bias=False,
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.mode == "channel":
            max_out = self.mlp(self.max_pool(x))
            avg_out = self.mlp(self.avg_pool(x))
            channel_out = self.sigmoid(max_out + avg_out)
            x = channel_out * x
        elif self.mode == "spatial":
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            avg_out = torch.mean(x, dim=1, keepdim=True)
            spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
            x = spatial_out * x
        elif self.mode == "cbam":
            max_out = self.mlp(self.max_pool(x))
            avg_out = self.mlp(self.avg_pool(x))
            channel_out = self.sigmoid(max_out + avg_out)
            x = channel_out * x
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            avg_out = torch.mean(x, dim=1, keepdim=True)
            spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
            x = spatial_out * x
        return x


class SFTLayer(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        # gamma
        self.gamma = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding="same"),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding="same"),
        )
        # beta
        self.beta = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding="same"),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding="same"),
        )

    def forward(self, x, cond):
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        return gamma * x + beta


class ConditionFusion(nn.Module):

    def __init__(
            self, in_channels, detail_channels, grad_channels, inter_channels, out_channels
    ):
        super().__init__()
        # detail
        self.detail = nn.Sequential(nn.Conv2d(detail_channels, inter_channels, 1, bias=False))
        # input
        self.input = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False))
        # condition map
        self.cond = nn.Sequential(
            nn.Conv2d(grad_channels, inter_channels, 1, padding="same", bias=False),
            nn.PReLU(),
            nn.Conv2d(inter_channels, inter_channels, 7, padding="same"),
            nn.PReLU(),
            nn.Conv2d(inter_channels, inter_channels, 3, padding="same"),
            nn.PReLU(),
        )

        # gamma
        self.gamma = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding="same"),
            nn.PReLU(),
            nn.Conv2d(inter_channels, inter_channels, 3, padding="same"),
        )
        # beta
        self.beta = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding="same"),
            nn.PReLU(),
            nn.Conv2d(inter_channels, inter_channels, 3, padding="same"),
        )
        # refine
        self.refine = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding="same"),
            nn.PReLU(),
            nn.Conv2d(inter_channels, inter_channels, 3, padding="same"),
        )

        self.final = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, 1, padding="same"),
        )

    def forward(self, input, x, grad):
        x = self.detail(x)
        input = self.input(input)

        grad = self.cond(grad)
        gamma = self.gamma(grad)
        beta = self.beta(grad)
        out = gamma * input + beta
        out = self.refine(out)

        out = out + x
        out = self.final(out)
        return out


_gradient_x = torch.tensor([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

_gradient_y = _gradient_x.detach().clone().T

_gradient_45 = torch.tensor([
    [2, 1, 0],
    [1, 0, -1],
    [0, -1, -2]
])

_gradient_n45 = torch.tensor([
    [0, 1, 2],
    [-1, 0, 1],
    [-2, -1, 0]
])

grad_kernel = torch.stack(
    [_gradient_x, _gradient_y, _gradient_45, _gradient_n45]
).reshape(-1, 1, 3, 3).to(torch.float32)


# 38.99
class SGN_with_Guide(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            inter_channels: int,
            padding_mode: str = "zeros",
            norm_op: Optional[Callable[..., nn.Module]] = None,
            act_op: Optional[Callable[..., nn.Module]] = nn.PReLU,
            device=None,
    ):
        super().__init__()
        self.base = SGNBase(in_channels, inter_channels,
                            padding_mode, norm_op, act_op)
        # Main stream
        self.main1 = Conv2dLayer(in_channels, inter_channels, 3,
                                 padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)
        self.main2 = Conv2dLayer(int(inter_channels * (2 ** 0 + 2 ** 1 / 4.0)), inter_channels, 3,
                                 padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)

        self.adj1 = SFTLayer(inter_channels)

        self.main3 = DenseResidualBlock(inter_channels, inter_channels, 3,
                                        padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)
        self.adj2 = SFTLayer(inter_channels)

        self.main4 = DenseResidualBlock(inter_channels, inter_channels, 3,
                                        padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)
        self.main5 = Conv2dLayer(inter_channels, out_channels, 3,
                                 padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)
        # Guide
        self.guide1 = nn.Sequential(
            Conv2dLayer(grad_kernel.shape[0], inter_channels, 3,
                        padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op),
            DenseResidualBlock(inter_channels, inter_channels, 3,
                               padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)
        )
        self.guide2 = Conv2dLayer(int(inter_channels * (2 ** 0 + 2 ** 1 / 4.0)), inter_channels, 3,
                                  padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)
        self.guide3 = nn.Sequential(
            DenseResidualBlock(inter_channels, inter_channels, 3,
                               padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op),
            Attention(inter_channels, 8),
        )
        self.guide4 = nn.Sequential(
            DenseResidualBlock(inter_channels, inter_channels, 3,
                               padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op),
            Attention(inter_channels, 8),
        )
        self.guide5 = Conv2dLayer(inter_channels, out_channels, 3,
                                  padding="same", padding_mode=padding_mode, norm_op=norm_op, act_op=act_op)


    def forward(self, x, gradient_x):
        base = self.base(x)  # inter_channel // 2

        guide = self.guide1(gradient_x)
        # guide = gradient_x
        guide = torch.cat((guide, base), dim=1)
        guide = self.guide2(guide)
        guide1 = self.guide3(guide) + guide
        guide2 = self.guide4(guide1) + guide1
        guide = self.guide5(guide2)

        out1 = self.main1(x)
        out1 = torch.cat((out1, base), dim=1)
        out1 = self.main2(out1)
        out1 = self.adj1(out1, guide1)
        out1 = self.main3(out1)
        out1 = self.adj2(out1, guide2)
        out1 = self.main4(out1)
        out1 = self.main5(out1)

        return out1, guide


class GGDBN(nn.Module):

    def __init__(
            self,
            model,
            device=None,
    ) -> None:
        super().__init__()
        super().__init__()
        self.model = model
        self.to_gray = Grayscale()
        self.gradient_kernel = grad_kernel.reshape(-1, 1, 3, 3).to(
            device=device, dtype=torch.float32
        )

        self.fusion = ConditionFusion(3, 3, 1, 64, 3)

        self.grad_gen = nn.Conv2d(3, 1, 3, padding="same")

    def forward(self, x):
        gray = self.to_gray(x)
        gray = F.conv2d(gray, self.gradient_kernel, padding=1)
        out1, guide = self.model(x, gray)

        grad = self.grad_gen(guide)

        out = self.fusion(x, out1, grad) + x

        return out, grad
