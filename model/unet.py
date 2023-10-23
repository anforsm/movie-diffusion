import torch
import torch.nn as nn
from collections import defaultdict

str_to_act = defaultdict(lambda: nn.ReLU())
str_to_act.update({
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
})


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        t = t.unsqueeze(-1)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        sin_enc = torch.sin(t.repeat(1, self.dim // 2) * inv_freq)
        cos_enc = torch.cos(t.repeat(1, self.dim // 2) * inv_freq)
        pos_enc = torch.cat([sin_enc, cos_enc], dim=-1)
        return pos_enc


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act="relu"):
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

        self.act = str_to_act[act]

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class EmbeddingBlock(nn.Module):
    def __init__(self, channels: int, emb_dim: int, act="relu"):
        super().__init__()

        self.lin = nn.Linear(emb_dim, channels)
        self.act = str_to_act[act]
    
    def forward(self, x):
        x = self.lin(x)
        x = self.act(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, channels: int, emb_dim: int, dropout: float = 0, out_channels=None):
        """A resblock with a time embedding and an optional change in channel count
        """
        if out_channels is None:
            out_channels = channels
        super().__init__()

        self.conv1 = ConvBlock(channels, out_channels)
        
        self.emb = EmbeddingBlock(out_channels) 

        self.conv2 = ConvBlock(out_channels, out_channels)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, t):
        x = self.conv1(x)

        t = self.emb(t)
        # t: (batch_size, time_embedding_dim) = (batch_size, out_channels)
        # x: (batch_size, out_channels, height, width)
        # we repeat the time embedding to match the shape of x
        t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.shape[2], x.shape[3])

        x = x + t

        x = self.conv2(x)
        x = self.dropout(x)
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=4,
            dropout=0,
            activation="relu",
            batch_first=True,
        )
    
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, -1).swapaxes(1, 2)
        x = self.transformer(x)
        return x.swapaxes(1, 2).view(batch_size, channels, height, width)


class DownBlock(nn.Module):
    """According to U-Net paper

    'The contracting path follows the typical architecture of a convolutional network.
    It consists of the repeated application of two 3x3 convolutions (unpadded convolutions),
    each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2
    for downsampling. At each downsampling step we double the number of feature channels.'
    """

    def __init__(self, in_channels, out_channels, time_embedding_dim, use_attn=False, dropout=0):
        """in_channels will typically be half of out_channels"""
        super().__init__()
        self.use_attn = use_attn

        self.resblock = ResBlock(
            channels=in_channels,
            out_channels=out_channels,
            emb_dim=time_embedding_dim,
            dropout=dropout,
        )

        self.downsample = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        if self.use_attn:
            self.attn = SelfAttentionBlock(
                channels=out_channels,
            )

    def forward(self, x, t):
        x = self.conv1(x)
        residual = x
        x = self.downsample(x)
        t = self.time_embedding_layer(t)

        x = x + t

        x = self.dropout(x)
        x = self.conv2(x)

        if self.use_attn:
            x = self.attn(x)
        return x, residual


class UpBlock(nn.Module):
    """According to U-Net paper

    Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2
    convolution (“up-convolution”) that halves the number of feature channels, a concatenation with
    the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions,
    each followed by a ReLU.
    """

    def __init__(self, in_channels, out_channels, time_embedding_dim, use_attn=False, dropout=0):
        """in_channels will typically be double of out_channels
        """
        super().__init__()
        self.use_attn = use_attn

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

        self.dropout = nn.Dropout(dropout)

        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
        )

        if self.use_attn:
            self.attn = SelfAttentionBlock(
                channels=out_channels,
            )

    def forward(self, x, t, residual=None):
        x = self.upsample(x)
        x = self.up_conv(x)

        if residual is not None:
            x = torch.cat([x, residual], dim=1)

        x = self.conv1(x)

        t = self.time_embedding_layer(t)
        # t: (batch_size, time_embedding_dim) = (batch_size, out_channels)
        # x: (batch_size, out_channels, height, width)
        # we repeat the time embedding to match the shape of x
        t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.shape[2], x.shape[3])
        x = x + t

        x = self.dropout(x)
        x = self.conv2(x)

        if self.use_attn:
            x = self.attn(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, time_embedding_dim):
        super().__init__()
        self.time_embedding_layer = nn.Sequential(
            nn.Linear(time_embedding_dim, out_channels),
            nn.ReLU(),
        )

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
        )
    
    def forward(self, x, t):
        x = self.conv1(x)
        
        t = self.time_embedding_layer(t)
        # t: (batch_size, time_embedding_dim) = (batch_size, out_channels)
        # x: (batch_size, out_channels, height, width)
        # we repeat the time embedding to match the shape of x
        t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.shape[2], x.shape[3])
        x = x + t

        x = self.dropout(x)
        x = self.conv2(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, channels, dropout, time_embedding_dim):
        super().__init__()
        self.channels = channels
        in_channels = channels
        out_channels = channels * 2
        self.conv1 = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            time_embedding_dim=time_embedding_dim
        )
        self.attn = SelfAttentionBlock(
            channels=out_channels,
        )
        self.conv2 = DoubleConv(
            in_channels=out_channels,
            out_channels=out_channels,
            dropout=dropout,
            time_embedding_dim=time_embedding_dim
        ) 
    
    def forward(self, x, t):
        x = self.conv1(x, t)
        x = self.attn(x)
        x = self.conv2(x, t)
        return x


class Unet(nn.Module):
    def __init__(
        self,
        image_channels,
        time_embedding_dim=128,
        dropout=0,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.time_encoding = SinusoidalPositionalEncoding(dim=time_embedding_dim)

        # Wide U-Net, i.e. num channels are increased as we celntract
        starting_channels = 64
        C = starting_channels
        channels_list_down = [
            # (image_channels, starting_channels),
            (3, C),
            (C, 2*C),
            (2*C, 4*C),
            (4*C, 8*C),
            #(512, 1024),
            #(starting_channels * 4, starting_channels * 8),
        ]
        # image size
        # 192 x 128
        # 96 x 64
        # 48 x 32
        # 24 x 16
        # 12 x 8

        # 32 x 32
        # 16 x 16
        # 8 x 8
        # 4 x 4

        channels_list_up = [
            #(starting_channels * 8 * 2, starting_channels * 4 * 2),
            #(2048, 1024),
            (16*C, 8*C),
            (8*C, 4*C),
            (4*C, 2*C),
            (2*C, C),
        ]

        use_attn = [
            False,
            False, 
            True,
            True,
        ]

        self.contracting_path = nn.ModuleList(
            [
                DownBlock(in_channels=in_channels, out_channels=out_channels, time_embedding_dim=time_embedding_dim, use_attn=attn, dropout=dropout)
                for (in_channels, out_channels), attn in zip(channels_list_down, use_attn)
            ]
        )

        self.bottleneck = Bottleneck(channels=channels_list_down[-1][-1], time_embedding_dim=time_embedding_dim, dropout=dropout) 

        self.expansive_path = nn.ModuleList(
            [
                # multiply by 2 since we concatenate the channels from the contracting path
                # also because of bottleneck doubling the channels
                UpBlock(in_channels=in_channels, out_channels=out_channels, time_embedding_dim=time_embedding_dim, dropout=dropout)
                for (in_channels, out_channels), attn in zip(channels_list_up, reversed(use_attn))
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
        # If using torchvision totensor, we do not need to swap axes
        # x: (batch_size, height, width, channels)
        #x = torch.einsum("bhwc->bchw", x)
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

        x = self.bottleneck(x, t)

        for expansive_block, residual in zip(
            self.expansive_path, reversed(contracting_residuals)
        ):
            x = expansive_block(x, t, residual)

        x = self.head(x)
        # x: (1, 3, 120, 80)
        #x = torch.einsum("bchw->bhwc", x)
        # x: (1, 120, 80, 3)
        return x
