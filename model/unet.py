from typing import Any
import torch
import torch.nn as nn
from collections import defaultdict
import torch as th
import numpy as np
import math

str_to_act = defaultdict(lambda: nn.SiLU())
str_to_act.update({
    "relu": nn.SiLU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
})

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

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
    def __init__(self, in_channels, out_channels, act="relu", dropout=None, zero=False):
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
            self.conv = zero_module(self.conv)

        
    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.conv(x)
        return x

class EmbeddingBlock(nn.Module):
    def __init__(self, channels: int, emb_dim: int, act="relu"):
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

        #self.dropout = nn.Dropout(dropout)
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

        #x = self.dropout(x)
        x = self.conv2(x)
        x = x + self.skip_connection(original)
        return x

class SelfAttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=4, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = nn.GroupNorm(32, channels)
        # not sure why openai is using conv1d instead of linear layer
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        #self.qkv = nn.Linear(channels, 3 * channels)
        self.attention = QKVAttention()
        self.proj_out = zero_module(
            # again, not sure why openai is using kernel 1 conv1d
            nn.Conv1d(channels, channels, 1)
            #nn.Linear(channels, channels),
        )

    def forward(self, x):
        b, c, *spatial = x.shape
        # batch size, channel dim, width, height
        x = x.reshape(b, c, -1)
        # we want to convert 2d image to 1d sequence
        # x: batch size, channel dim, width * height
        qkv = self.qkv(self.norm(x))
        # get Q, K, V matrices

        # qkv: batch size, channel dim * 3, width * height ?
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        # qkv: batch size * num heads, channel dim * 3, width * height ?

        # simple attn operation: attn = softmax(qk^T / sqrt(d)) v
        h = self.attention(qkv)

        # h: batch size * num heads, width * height, channel dim ?
        h = h.reshape(b, -1, h.shape[-1])
        # h: batch size, width * height, channel dim * num heads ?
        h = self.proj_out(h)
        # uses an additory residual connection
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

class SelfAttentionBlockOld(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=1,
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

    def __init__(self, in_channels, out_channels, time_embedding_dim, use_attn=False, dropout=0, downsample=True, width=1):
        """in_channels will typically be half of out_channels"""
        super().__init__()
        self.width = width
        self.use_attn = use_attn
        self.do_downsample = downsample

        #print(f"DownBlock: {in_channels} -> {out_channels}")

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

        #self.resblock = ResBlock(
        #    channels=in_channels,
        #    out_channels=out_channels,
        #    emb_dim=time_embedding_dim,
        #    dropout=dropout,
        #)

        #if self.use_attn:
        #    self.attn = SelfAttentionBlock(
        #        channels=out_channels,
        #    )

        if self.do_downsample:
            self.downsample = Downsample(out_channels) 

    def forward(self, x, t):
        #x = self.resblock(x, t)

        #if self.use_attn:
        #    x = self.attn(x)
        for block in self.blocks:
            if isinstance(block, ResBlock):
                x = block(x, t)
            elif isinstance(block, SelfAttentionBlock):
                x = block(x)

        if self.do_downsample:
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

    def __init__(self, in_channels, out_channels, time_embedding_dim, use_attn=False, dropout=0, upsample=True, width=1):
        """in_channels will typically be double of out_channels
        """
        super().__init__()
        self.use_attn = use_attn
        self.do_upsample = upsample

        #print(f"UpBlock: {in_channels} -> {out_channels}")

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

        #self.resblock = ResBlock(
        #    channels=in_channels,
        #    out_channels=out_channels,
        #    emb_dim=time_embedding_dim,
        #    dropout=dropout,
        #)

        #if self.use_attn:
        #    self.attn = SelfAttentionBlock(
        #        channels=out_channels,
        #    )
        
        if self.do_upsample:
            self.upsample = Upsample(out_channels)

    def forward(self, x, t, residual=None):
        #x = self.resblock(x, t)

        #if self.use_attn:
        #    x = self.attn(x)
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
        image_channels,
        time_embedding_dim=128,
        dropout=0,
    ):
        super().__init__()
        res_block_width = 2
        starting_channels = 128
        time_embedding_dim = 4 * starting_channels
        C = starting_channels

        self.image_channels = image_channels
        self.time_encoding = SinusoidalPositionalEncoding(dim=C)
        self.time_embedding = TimeEmbedding(model_dim=C, emb_dim=time_embedding_dim)

        #self.input = ConvBlock(3, C)

        #channel_mults = (1, 2, 4, 8)
        #channel_mults = (1, 2, 4, 8)
        #channel_mults = (1, 2)
        channel_mults = (1, 2, 2, 4, 4)

        # Wide U-Net, i.e. num channels are increased as we celntract

        use_attn = [
            False,
            False, 
            False,
            #False, 
            #True,
            #True,
            True,
            False,
        ]

        self.contracting_path = nn.ModuleList(
            [
                nn.Conv2d(3, C, kernel_size=3, padding=1)
            ]
        )
        input_chs = []
        ch = C
        dimensions = 1
        for i, mult in enumerate(channel_mults):
            self.contracting_path.append(DownBlock(
                in_channels=ch,
                out_channels=mult * C,
                time_embedding_dim=time_embedding_dim,
                use_attn=use_attn[i],
                dropout=dropout,
                downsample=False,
                width=res_block_width,
            ))
            ch = mult * C
            input_chs.append(ch)
            if i != len(channel_mults) - 1:
                self.contracting_path.append(Downsample(ch))
                dimensions *= 2

        #for i, ((in_channels, out_channels), attn) in enumerate(zip(channels_list_down, use_attn)):
        #    self.contracting_path.append(DownBlock(in_channels=in_channels, out_channels=out_channels, time_embedding_dim=time_embedding_dim, use_attn=attn, dropout=dropout, downsample=False, width=res_block_width))
        #    if i != len(channels_list_down) - 1:
        #        self.contracting_path.append(Downsample(out_channels))

        #self.bottleneck = Bottleneck(channels=channels_list_down[-1][-1], time_embedding_dim=time_embedding_dim, dropout=dropout) 
        self.bottleneck = Bottleneck(channels=ch, time_embedding_dim=time_embedding_dim, dropout=dropout) 
        #print("input chs", input_chs)

        self.expansive_path = nn.ModuleList(
            [
                # multiply by 2 since we concatenate the channels from the contracting path
                # also because of bottleneck doubling the channels
                #UpBlock(in_channels=in_channels, out_channels=out_channels, time_embedding_dim=time_embedding_dim, use_attn=attn, dropout=dropout, upsample=i != 0)
                #for i, ((in_channels, out_channels), attn) in enumerate(zip(channels_list_up, reversed(use_attn)))

                # UpBlock(in_channels=in_channels, out_channels=out_channels, time_embedding_dim=time_embedding_dim, use_attn=False, dropout=dropout, upsample=i != 0 and i % 2 == 1 and i != len(channels_list_up) - 1)
                # for i, (in_channels, out_channels) in enumerate(channels_list_up)
            ]
        )
        for i, mult in enumerate(reversed(channel_mults)):
            self.expansive_path.append(UpBlock(
                in_channels=ch + input_chs.pop(),
                out_channels=mult * C,
                time_embedding_dim=time_embedding_dim,
                use_attn=list(reversed(use_attn))[i],
                dropout=dropout,
                #upsample=i != 0 and i != len(channel_mults) - 1,# and i % 2 == 1,# and i != len(channel_mults) - 1,
                #upsample=i != 0,
                upsample=i != len(channel_mults) - 1,
                width=res_block_width,
            ))
            ch = mult * C

        self.head = nn.Sequential(
            nn.GroupNorm(32, starting_channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(
                in_channels=starting_channels,
                out_channels=image_channels,
                kernel_size=3,
                padding=1,
            ))
        )

    def forward(self, x, t):
        # If using torchvision totensor, we do not need to swap axes
        # x: (batch_size, height, width, channels)
        #x = torch.einsum("bhwc->bchw", x)
        # x: (batch_size, channels, height, width)
        t = self.time_encoding(t)
        t = self.time_embedding(t)

        #x = self.input(x)

        residuals = []
        # x: (1, 3, 120, 80)
        # x: (1, 64, 120, 80)
        # x: (1, 128, 60, 40)
        # x: (1, 256, 30, 20)
        # x: (1, 512, 15, 10)
        for contracting_block in self.contracting_path:
            if isinstance(contracting_block, Downsample) or isinstance(contracting_block, nn.Conv2d):
                x = contracting_block(x)
            else:
                x = contracting_block(x, t)
                #print(x.shape)
                residuals.append(x)

        x = self.bottleneck(x, t)
        #print("going up")
        #print([r.shape for r in residuals])

        #for expansive_block, residual in zip(
        #    self.expansive_path, reversed(residuals)
        #):
        for expansive_block in self.expansive_path:
            if isinstance(expansive_block, UpBlock):
                residual = residuals.pop()
                #print(f"residual shape: {residual.shape}")
                #print(f"expansive_block shape: {x.shape}")
                x = torch.cat([x, residual], dim=1)
                x = expansive_block(x, t)
            else:
                x = expansive_block(x, t)

        x = self.head(x)
        # x: (1, 3, 120, 80)
        #x = torch.einsum("bchw->bhwc", x)
        # x: (1, 120, 80, 3)
        #print("last", x.shape)
        return x
