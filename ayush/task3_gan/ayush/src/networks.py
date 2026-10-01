"""CycleGAN generator (ResNet-9) and 70x70 PatchGAN. No pretrained weights."""

from __future__ import annotations

import random

import torch
import torch.nn as nn


def _norm(channels: int) -> nn.InstanceNorm2d:
    return nn.InstanceNorm2d(channels, affine=True, track_running_stats=False)


class ResnetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            _norm(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            _norm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """c7s1-64, d128, d256, 9 residual blocks, u128, u64, c7s1-3 tanh."""

    def __init__(self, n_blocks: int = 9):
        super().__init__()
        if n_blocks != 9:
            raise ValueError("this lab generator is the 9-block 256 setup")
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, bias=False),
            _norm(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            _norm(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            _norm(256),
            nn.ReLU(True),
        ]
        layers += [ResnetBlock(256) for _ in range(n_blocks)]
        layers += [
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=False),
            _norm(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=False),
            _norm(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN. On 256 input this returns a 30x30 score map, no sigmoid."""

    def __init__(self):
        super().__init__()
        def down(cin, cout, stride, norm):
            block: list[nn.Module] = [nn.Conv2d(cin, cout, 4, stride=stride, padding=1, bias=not norm)]
            if norm:
                block.append(_norm(cout))
            block.append(nn.LeakyReLU(0.2, True))
            return block

        layers: list[nn.Module] = []
        layers += down(3, 64, 2, False)
        layers += down(64, 128, 2, True)
        layers += down(128, 256, 2, True)
        # Last wide layer is stride 1 so a 256 image maps to 30x30, not 15x15.
        layers += down(256, 512, 1, True)
        layers += [nn.Conv2d(512, 1, 4, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImagePool:
    def __init__(self, size: int = 50):
        self.size = size
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        out = []
        for img in images:
            img = img.detach()
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif random.random() > 0.5:
                ix = random.randrange(len(self.images))
                old = self.images[ix]
                self.images[ix] = img
                out.append(old)
            else:
                out.append(img)
        return torch.stack(out)


def init_weights(module: nn.Module) -> None:
    """Conv weights ~ N(0, 0.02). InstanceNorm affine starts at 1, or the first block is zero."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)
