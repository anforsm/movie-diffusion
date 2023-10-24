from typing import Any
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

class TimeEmbedding(nn.Module):
    def __init__(self, model_dim: int, emb_dim: int, act="relu"):
        super().__init__()

        self.lin = nn.Linear(model_dim, emb_dim)
        self.act = str_to_act[act]
        self.lin2 = nn.Linear(emb_dim, emb_dim)
    
    def forward(self, x):
        x = self.lin(x)
        x = self.act(x)
        x = self.lin2(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act="relu"):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        # Add GroupNorm according to DDPM paper
        self.norm = nn.GroupNorm(
            num_groups=32,
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
        
        self.emb = EmbeddingBlock(out_channels, emb_dim) 

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

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # ddpm uses maxpool
        # self.down = nn.MaxPool2d

        # iddpm uses strided conv
        self.down = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
    
    def forward(self, x):
        return self.down(x)


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

        if self.use_attn:
            self.attn = SelfAttentionBlock(
                channels=out_channels,
            )

        self.downsample = Downsample(out_channels) 

    def forward(self, x, t):
        x = self.resblock(x, t)

        if self.use_attn:
            x = self.attn(x)

        x = self.downsample(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

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

        self.resblock = ResBlock(
            channels=in_channels,
            out_channels=out_channels,
            emb_dim=time_embedding_dim,
            dropout=dropout,
        )

        if self.use_attn:
            self.attn = SelfAttentionBlock(
                channels=out_channels,
            )
        
        self.upsample = Upsample(out_channels)

    def forward(self, x, t, residual=None):
        x = self.resblock(x, t)

        if self.use_attn:
            x = self.attn(x)

        x = self.upsample(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, channels, dropout, time_embedding_dim):
        super().__init__()
        in_channels = channels
        out_channels = channels
        self.resblock_1 = ResBlock(
            channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            emb_dim=time_embedding_dim
        )
        self.attention_block = SelfAttentionBlock(
            channels=out_channels,
        )
        self.resblock_2 = ResBlock(
            channels=out_channels,
            out_channels=out_channels,
            dropout=dropout,
            emb_dim=time_embedding_dim
        ) 
    
    def forward(self, x, t):
        x = self.resblock_1(x, t)
        x = self.attention_block(x)
        x = self.resblock_2(x, t)
        return x

class Unet(nn.Module):
    def __init__(
        self,
        image_channels,
        time_embedding_dim=128,
        dropout=0,
    ):
        super().__init__()
        starting_channels = 64
        C = starting_channels

        self.image_channels = image_channels
        self.time_encoding = SinusoidalPositionalEncoding(dim=C)
        self.time_embedding = TimeEmbedding(model_dim=C, emb_dim=time_embedding_dim)

        self.input = ConvBlock(3, C)

        # Wide U-Net, i.e. num channels are increased as we celntract
        channels_list_down = [
            # (image_channels, starting_channels),
            (C, C),
            (C, 2*C),
            (2*C, 4*C),
            (4*C, 8*C),
            #(8*C, 16*C),
        ]

        channels_list_up = [
            #(16*C, 8*C),
            (8*C, 4*C),
            (4*C, 2*C),
            (2*C, C),
            (C, C),
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
                UpBlock(in_channels=in_channels + in_channels, out_channels=out_channels, time_embedding_dim=time_embedding_dim, use_attn=attn, dropout=dropout)
                for (in_channels, out_channels), attn in zip(channels_list_up, reversed(use_attn))
            ]
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=starting_channels,
                out_channels=image_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, x, t):
        # If using torchvision totensor, we do not need to swap axes
        # x: (batch_size, height, width, channels)
        #x = torch.einsum("bhwc->bchw", x)
        # x: (batch_size, channels, height, width)
        t = self.time_encoding(t)
        t = self.time_embedding(t)

        x = self.input(x)

        residuals = []
        # x: (1, 3, 120, 80)
        # x: (1, 64, 120, 80)
        # x: (1, 128, 60, 40)
        # x: (1, 256, 30, 20)
        # x: (1, 512, 15, 10)
        for contracting_block in self.contracting_path:
            x = contracting_block(x, t)
            residuals.append(x)

        x = self.bottleneck(x, t)

        #print([x.shape for x in residuals])
        for expansive_block, residual in zip(
            self.expansive_path, reversed(residuals)
        ):
            x = torch.cat([x, residual], dim=1)
            x = expansive_block(x, t)

        x = self.head(x)
        # x: (1, 3, 120, 80)
        #x = torch.einsum("bchw->bhwc", x)
        # x: (1, 120, 80, 3)
        return x
