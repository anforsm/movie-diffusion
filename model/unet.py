from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import defaultdict
import torch as th
import numpy as np
import math

str_to_act = defaultdict(lambda: nn.SiLU())
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
    def __init__(self, model_dim: int, emb_dim: int, act="silu"):
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
    def __init__(self, in_channels, out_channels, act="silu", dropout=None, zero=False):
        super().__init__()

        self.norm = nn.GroupNorm(
            num_groups=32,
            num_channels=in_channels,
        )

        self.act = str_to_act[act]

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        if zero:
            self.conv.weight.data.zero_()

        
    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.conv(x)
        return x

class EmbeddingBlock(nn.Module):
    def __init__(self, channels: int, emb_dim: int, act="silu"):
        super().__init__()

        self.act = str_to_act[act]
        self.lin = nn.Linear(emb_dim, channels)
    
    def forward(self, x):
        x = self.act(x)
        x = self.lin(x)
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

        self.conv2 = ConvBlock(out_channels, out_channels, dropout=dropout, zero=True)

        if channels != out_channels:
            self.skip_connection = nn.Conv2d(channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()

    
    def forward(self, x, t):
        original = x
        x = self.conv1(x)

        t = self.emb(t)
        # t: (batch_size, time_embedding_dim) = (batch_size, out_channels)
        # x: (batch_size, out_channels, height, width)
        # we repeat the time embedding to match the shape of x
        t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.shape[2], x.shape[3])

        x = x + t

        x = self.conv2(x)
        x = x# + self.skip_connection(original)
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(32, channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=0,
            batch_first=True,
            bias=True,
        )
    
    def forward(self, x):
        h, w = x.shape[-2:]
        original = x
        x = self.norm(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.attention(x, x, x)[0]
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x + original

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

    def __init__(self, in_channels, out_channels, time_embedding_dim, use_attn=False, dropout=0, downsample=True, width=1):
        """in_channels will typically be half of out_channels"""
        super().__init__()
        self.width = width
        self.use_attn = use_attn
        self.do_downsample = downsample

        self.blocks = nn.ModuleList()
        for _ in range(width):
            self.blocks.append(ResBlock(
                channels=in_channels,
                out_channels=out_channels,
                emb_dim=time_embedding_dim,
                dropout=dropout,
            ))
            if self.use_attn:
                self.blocks.append(SelfAttentionBlock(
                    channels=out_channels,
                ))
            in_channels = out_channels

        if self.do_downsample:
            self.downsample = Downsample(out_channels) 

    def forward(self, x, t):
        for block in self.blocks:
            if isinstance(block, ResBlock):
                x = block(x, t)
            elif isinstance(block, SelfAttentionBlock):
                x = block(x)

        residual = x
        if self.do_downsample:
            x = self.downsample(x)
        return x, residual

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

    def __init__(self, in_channels, out_channels, time_embedding_dim, use_attn=False, dropout=0, upsample=True, width=1):
        """in_channels will typically be double of out_channels
        """
        super().__init__()
        self.use_attn = use_attn
        self.do_upsample = upsample

        self.blocks = nn.ModuleList()
        for _ in range(width):
            self.blocks.append(ResBlock(
                channels=in_channels,
                out_channels=out_channels,
                emb_dim=time_embedding_dim,
                dropout=dropout,
            ))
            if self.use_attn:
                self.blocks.append(SelfAttentionBlock(
                    channels=out_channels,
                ))
            in_channels = out_channels

        if self.do_upsample:
            self.upsample = Upsample(out_channels)

    def forward(self, x, t):
        for block in self.blocks:
            if isinstance(block, ResBlock):
                x = block(x, t)
            elif isinstance(block, SelfAttentionBlock):
                x = block(x)

        if self.do_upsample:
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
        image_channels=3,
        res_block_width=2,
        starting_channels=128,
        dropout=0,
        channel_mults=(1, 2, 2, 4, 4),
        attention_layers=(False, False, False, True, False)
    ):
        super().__init__()
        self.is_conditional = False

        self.image_channels = image_channels
        self.starting_channels = starting_channels
        time_embedding_dim = 4 * starting_channels

        self.time_encoding = SinusoidalPositionalEncoding(dim=starting_channels)
        self.time_embedding = TimeEmbedding(model_dim=starting_channels, emb_dim=time_embedding_dim)

        self.input = nn.Conv2d(3, starting_channels, kernel_size=3, padding=1)

        current_channel_count = starting_channels

        input_channel_counts = []
        self.contracting_path = nn.ModuleList([])
        for i, channel_multiplier in enumerate(channel_mults):
            is_last_layer = i == len(channel_mults) - 1
            next_channel_count = channel_multiplier * starting_channels

            self.contracting_path.append(DownBlock(
                in_channels=current_channel_count,
                out_channels=next_channel_count,
                time_embedding_dim=time_embedding_dim,
                use_attn=attention_layers[i],
                dropout=dropout,
                downsample=not is_last_layer,
                width=res_block_width,
            ))
            current_channel_count = next_channel_count 

            input_channel_counts.append(current_channel_count)

        self.bottleneck = Bottleneck(channels=current_channel_count, time_embedding_dim=time_embedding_dim, dropout=dropout) 

        self.expansive_path = nn.ModuleList([])
        for i, channel_multiplier in enumerate(reversed(channel_mults)):
            next_channel_count = channel_multiplier * starting_channels

            self.expansive_path.append(UpBlock(
                in_channels=current_channel_count + input_channel_counts.pop(),
                out_channels=next_channel_count,
                time_embedding_dim=time_embedding_dim,
                use_attn=list(reversed(attention_layers))[i],
                dropout=dropout,
                upsample=i != len(channel_mults) - 1,
                width=res_block_width,
            ))
            current_channel_count = next_channel_count

        last_conv = nn.Conv2d(
            in_channels=starting_channels,
            out_channels=image_channels,
            kernel_size=3,
            padding=1,
        )
        last_conv.weight.data.zero_()

        self.head = nn.Sequential(
            nn.GroupNorm(32, starting_channels),
            nn.SiLU(),
            last_conv,
        )

    def forward(self, x, t):
        t = self.time_encoding(t)
        return self._forward(x, t)

    def _forward(self, x, t):
        t = self.time_embedding(t)

        x = self.input(x)

        residuals = []
        for contracting_block in self.contracting_path:
            x, residual = contracting_block(x, t)
            residuals.append(residual)

        x = self.bottleneck(x, t)

        for expansive_block in self.expansive_path:
            # Add the residual
            residual = residuals.pop()
            x = torch.cat([x, residual], dim=1)

            x = expansive_block(x, t)

        x = self.head(x)
        return x

class ConditionalUnet(nn.Module):
    def __init__(self, unet, num_classes):
        super().__init__()
        self.is_conditional = True

        self.unet = unet
        self.num_classes = num_classes

        self.class_embedding = nn.Embedding(num_classes, unet.starting_channels)
    
    def forward(self, x, t, cond=None):
        # cond: (batch_size, n), where n is the number of classes that we are conditioning on
        t = self.unet.time_encoding(t)

        if cond is not None:
            cond = self.class_embedding(cond)
            # sum across the classes so we get a single vector representing the set of classes
            cond = cond.sum(dim=1)
            t += cond

        return self.unet._forward(x, t)