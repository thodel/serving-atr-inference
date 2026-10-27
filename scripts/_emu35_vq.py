# ruff: noqa  -- vendored upstream code, kept as BAAI wrote it
# Vendored from BAAI/Emu3.5-VisionTokenizer, modeling_emu3p5visionvq.py
# (https://huggingface.co/BAAI/Emu3.5-VisionTokenizer, Apache-2.0):
#
#   Copyright 2025 The Emu team, BAAI and The HuggingFace Inc. team. All rights reserved.
#   Licensed under the Apache License, Version 2.0 (the "License"); you may not use
#   this file except in compliance with the License. You may obtain a copy of the
#   License at http://www.apache.org/licenses/LICENSE-2.0
#
# Changes: the transformers PreTrainedModel/config wrappers are dropped and replaced
# by `Emu35VQ` below, so the weights load without trust_remote_code. The layers
# themselves are unchanged. Used only by scripts/apertus_vq_roundtrip.py: Apertus
# 1.5's vision tokenizer is an encode-only port of this one, and the decoder here
# is what turns its codes back into an image.

""" Emu3p5VisionVQ model """


import math
from typing import Optional, Tuple, Union

import torch
from torch import nn, einsum
from torch.nn import functional as F



def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Emu3p5VisionVQNormalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Emu3p5VisionVQUpsample(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Emu3p5VisionVQDownsample(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Emu3p5VisionVQResnetBlock(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Emu3p5VisionVQNormalize(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm2 = Emu3p5VisionVQNormalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Emu3p5VisionVQAttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Emu3p5VisionVQNormalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h * w) # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Emu3p5VisionVQEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.in_channels = config.in_channels
        self.resolution = config.resolution

        # downsampling
        self.conv_in = nn.Conv2d(
            self.in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        curr_res = self.resolution

        in_ch_mult = (1, ) + tuple(config.ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = config.ch * in_ch_mult[i_level]
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    Emu3p5VisionVQResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=config.dropout,
                    ),
                )
                block_in = block_out
                if curr_res in config.attn_resolutions:
                    attn.append(Emu3p5VisionVQAttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Emu3p5VisionVQDownsample(block_in)
                curr_res = curr_res // 2

            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = Emu3p5VisionVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
        )
        self.mid.attn_1 = Emu3p5VisionVQAttnBlock(block_in)
        self.mid.block_2 = Emu3p5VisionVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
        )

        # end
        self.norm_out = Emu3p5VisionVQNormalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * config.z_channels if config.double_z else config.z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )


    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Emu3p5VisionVQDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks

        self.resolution = config.resolution

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1, ) + tuple(config.ch_mult)
        block_in = config.ch * config.ch_mult[self.num_resolutions-1]

        curr_res = config.resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, config.z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(
            config.z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = Emu3p5VisionVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
        )
        self.mid.attn_1 = Emu3p5VisionVQAttnBlock(block_in)
        self.mid.block_2 = Emu3p5VisionVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    Emu3p5VisionVQResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=config.dropout,
                    ),
                )
                block_in = block_out
                if curr_res in config.attn_resolutions:
                    attn.append(Emu3p5VisionVQAttnBlock(block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Emu3p5VisionVQUpsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Emu3p5VisionVQNormalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            config.out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


class Emu3p5VisionVQVectorQuantizer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.n_e = config.codebook_size
        self.e_dim = config.embed_dim

        self.embedding = nn.Embedding(self.n_e, self.e_dim)

    def forward(self, z):
        # z: [b, d, h, w]
        embedding = self.embedding.weight  # [n, d]

        # cal similarity
        logits = torch.einsum("b d h w, n d -> b n h w", z, embedding)

        # get max indices
        ind = logits.argmax(dim=1)  # [b, h, w]

        # lookup embedding
        z_q = embedding[ind]  # [b, h, w, d]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # -> [b, d, h, w]

        return z_q, ind.flatten()

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)

        # shape should in B H W
        if shape is not None:
            if len(shape) == 3:
                shape = shape + (self.e_dim, )

            z_q = z_q.view(shape)

            # reshape back to match original input shape
            # b h w c -> b c h w
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class Emu35VQ(nn.Module):
    """Encoder, codebook and decoder with the checkpoint's own parameter names."""

    CONFIG = dict(attn_resolutions=[16], ch=256, ch_mult=[1, 1, 2, 2, 4], codebook_size=131072,
                  double_z=False, dropout=0.0, embed_dim=256, in_channels=3, num_res_blocks=4,
                  out_ch=3, resolution=256, z_channels=256)   # BAAI's config.json

    def __init__(self):
        super().__init__()
        from types import SimpleNamespace
        config = SimpleNamespace(**self.CONFIG)
        self.encoder = Emu3p5VisionVQEncoder(config)
        self.decoder = Emu3p5VisionVQDecoder(config)
        self.quantize = Emu3p5VisionVQVectorQuantizer(config)
        self.quant_conv = nn.Conv2d(config.z_channels, config.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.embed_dim, config.z_channels, 1)

    @torch.no_grad()
    def encode_codes(self, pixels: torch.Tensor) -> torch.Tensor:
        """(b, 3, h, w) in [-1, 1] -> (b, h/16, w/16) code indices."""
        h = self.quant_conv(self.encoder(pixels))
        logits = torch.einsum("bdhw,nd->bnhw", h, self.quantize.embedding.weight)
        return logits.argmax(dim=1)

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """(b, h/16, w/16) code indices -> (b, 3, h, w), roughly in [-1, 1]."""
        z = self.quantize.embedding(codes).permute(0, 3, 1, 2).contiguous()
        return self.decoder(self.post_quant_conv(z))
