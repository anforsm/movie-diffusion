import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        sin_enc = torch.sin(t.repeat(1, self.dim // 2) * inv_freq)
        cos_enc = torch.cos(t.repeat(1, self.dim // 2) * inv_freq)
        pos_enc = torch.cat([sin_enc, cos_enc], dim=-1)
        return pos_enc


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
        )

        # Add GroupNorm according to DDPM paper
        self.norm = nn.GroupNorm(
            num_groups=1,
            num_channels=out_channels,
        )

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

# Using Transformer and Encoder
class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, size: int):
        super().__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])

        # They use GELU?
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

    # DO NOT DO NORM BEFORE ATTENTION? CHANGE SWAP AXES
    def forward(self, x):
        # swapaxes
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        q = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)


class DownBlock(nn.Module):
    """According to U-Net paper

    'The contracting path follows the typical architecture of a convolutional network.
    It consists of the repeated application of two 3x3 convolutions (unpadded convolutions),
    each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2
    for downsampling. At each downsampling step we double the number of feature channels.'
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
        )

        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
        )

        self.downsample = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    """According to U-Net paper

    Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2
    convolution (“up-convolution”) that halves the number of feature channels, a concatenation with
    the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions,
    each followed by a ReLU.
    """

    def __init__(self, in_channels, out_channels, residual=None):
        super().__init__()

        self.upsample = nn.Upsample(in_channels, scale_factor=2)
        self.up_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
        )

        self.conv1 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
        )

        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.up_conv(x)

        if self.residual is not None:
            x = torch.cat([x, self.residual], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Unet(nn.Module):
    def __init__(
        self,
        image_channels,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.time_encoding = SinusoidalPositionalEncoding(dim=2)

        starting_channels = 64
        # Wide U-Net, i.e. num channels are increased as we celntract

        self.input_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=starting_channels,
                kernel_size=3,
            ),
            nn.ReLU(),
        )

        channels_list = [
            # (image_channels, starting_channels),
            (starting_channels, starting_channels * 2),
            (starting_channels * 2, starting_channels * 4),
            (starting_channels * 4, starting_channels * 8),
            (starting_channels * 8, starting_channels * 16),
            (starting_channels * 16, starting_channels * 32),
        ]

        self.contracting_path = nn.ModuleList(
            [
                DownBlock(in_channels=in_channels, out_channels=out_channels)
                for in_channels, out_channels in channels_list
            ]
        )

        self.bottleneck = lambda x: x

        self.expansive_path = nn.ModuleList(
            [
                UpBlock(in_channels=in_channels, out_channels=out_channels)
                for out_channels, in_channels in reversed(channels_list)
            ]
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=starting_channels,
                out_channels=image_channels,
                kernel_size=3,
            ),
            nn.ReLU(),
        )

    def forward(self, x, t):
        t = self.time_encoding(t)

        contracting_residuals = []
        x = self.input_layer(x)
        for contracting_block in self.contracting_path:
            x = contracting_block(x)
            contracting_residuals.append(x)

        x = self.bottleneck(x)

        for expansive_block, residual in zip(
            self.expansive_path, reversed(contracting_residuals)
        ):
            x = expansive_block(x, residual)

        x = self.output_layer(x)
        return x
