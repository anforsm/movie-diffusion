import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
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
            padding="same",
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

    def __init__(self, in_channels, out_channels, time_embedding_dim):
        """in_channels will typically be half of out_channels"""
        super().__init__()

        self.time_embedding_layer = nn.Sequential(
            nn.Linear(time_embedding_dim, out_channels),
            nn.ReLU(),
        )

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

    def forward(self, x, t):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = x
        x = self.downsample(x)
        t = self.time_embedding_layer(t)
        # t: (batch_size, time_embedding_dim) = (batch_size, out_channels)
        # x: (batch_size, out_channels, height, width)
        # we repeat the time embedding to match the shape of x
        t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.shape[2], x.shape[3])
        x = x + t 
        return x, residual


class UpBlock(nn.Module):
    """According to U-Net paper

    Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2
    convolution (“up-convolution”) that halves the number of feature channels, a concatenation with
    the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions,
    each followed by a ReLU.
    """

    def __init__(self, in_channels, out_channels, time_embedding_dim):
        """in_channels will typically be double of out_channels
        """
        super().__init__()

        self.time_embedding_layer = nn.Sequential(
            nn.Linear(time_embedding_dim, out_channels),
            nn.ReLU(),
        )

        self.upsample = nn.Upsample(scale_factor=2)
        self.up_conv = nn.Conv2d(
            # double the number of input channels, since we concatenate
            # the channels with those from the residual
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            padding="same"
        )

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
        )

        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
        )

    def forward(self, x, t, residual=None):
        x = self.upsample(x)
        x = self.up_conv(x)

        if residual is not None:
            x = torch.cat([x, residual], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        t = self.time_embedding_layer(t)
        # t: (batch_size, time_embedding_dim) = (batch_size, out_channels)
        # x: (batch_size, out_channels, height, width)
        # we repeat the time embedding to match the shape of x
        t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.shape[2], x.shape[3])
        x = x + t
        return x

class Bottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv1 = ConvBlock(
            in_channels=channels,
            out_channels=channels*2,
        )
        self.conv2 = ConvBlock( 
            in_channels=channels*2,
            out_channels=channels*2,
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Unet(nn.Module):
    def __init__(
        self,
        image_channels,
        time_embedding_dim=128,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.time_encoding = SinusoidalPositionalEncoding(dim=time_embedding_dim)

        # Wide U-Net, i.e. num channels are increased as we celntract
        starting_channels = 64
        channels_list_down = [
            # (image_channels, starting_channels),
            (3, 64),
            (64, 128),
            (128, 256),
            #(starting_channels * 4, starting_channels * 8),
        ]

        channels_list_up = [
            #(starting_channels * 8 * 2, starting_channels * 4 * 2),
            (512, 256),
            (256, 128),
            (128, 64),
        ]

        self.contracting_path = nn.ModuleList(
            [
                DownBlock(in_channels=in_channels, out_channels=out_channels, time_embedding_dim=time_embedding_dim)
                for in_channels, out_channels in channels_list_down
            ]
        )

        self.bottleneck = Bottleneck(channels=channels_list_down[-1][-1]) 

        self.expansive_path = nn.ModuleList(
            [
                # multiply by 2 since we concatenate the channels from the contracting path
                # also because of bottleneck doubling the channels
                UpBlock(in_channels=in_channels, out_channels=out_channels, time_embedding_dim=time_embedding_dim)
                for in_channels, out_channels in channels_list_up
            ]
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=starting_channels,
                out_channels=image_channels,
                kernel_size=1,
                padding="same",
            )
        )

    def forward(self, x, t):
        # x: (batch_size, height, width, channels)
        x = torch.einsum("bhwc->bchw", x)
        # x: (batch_size, channels, height, width)
        t = self.time_encoding(t)

        contracting_residuals = []
        # x: (1, 3, 120, 80)
        # x: (1, 64, 120, 80)
        # x: (1, 128, 60, 40)
        # x: (1, 256, 30, 20)
        # x: (1, 512, 15, 10)
        for contracting_block in self.contracting_path:
            x, residual = contracting_block(x, t)
            contracting_residuals.append(residual)

        x = self.bottleneck(x)

        for expansive_block, residual in zip(
            self.expansive_path, reversed(contracting_residuals)
        ):
            x = expansive_block(x, t, residual)

        x = self.head(x)
        # x: (1, 3, 120, 80)
        x = torch.einsum("bchw->bhwc", x)
        # x: (1, 120, 80, 3)
        return x
