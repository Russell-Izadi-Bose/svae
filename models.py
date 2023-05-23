#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Russell Izadi 2023
"""

import torch
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        attrs,
        rates,
        d_input,
        d_model,
        d_latent,
        n_heads,
        n_layers,
        max_duration,
    ):
        super().__init__()

        self.attrs = attrs
        self.rates = rates

        self.in_pos_emb = nn.Embedding(max_duration, d_model)
        self.out_pos_emb = nn.Embedding(max_duration, d_model)
        self.out_attr_emb = nn.Embedding(max_duration, d_model)

        self.in_layer = nn.Linear(d_input, d_model)
        self.out_layer = nn.Linear(d_model, d_latent)

        enc_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2048,
            dropout=0.1,
            activation=F.relu,
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
            device=None,
            dtype=None)
        self.enc_layers = TransformerDecoder(
            decoder_layer=enc_layer,
            num_layers=n_layers,
            norm=None)

    def forward(self, x, mask=None):

        in_duration = x.shape[1]

        # Input layer
        x = self.in_layer(x)
        print(x.shape)

        # Add positional embeddings
        in_pos = torch.LongTensor(list(range(in_duration))).to(x.device)
        x = x + self.in_pos_emb(in_pos)
        print(x.shape)

        # Init output with positional and attribute embeddings
        out_attr = []
        out_pos = []
        durations = []
        for i_attr, rate in enumerate(self.rates):
            duration_i = max(int(in_duration * rate), 1)
            durations.append(duration_i)
            out_attr.extend(duration_i * [i_attr])
            out_pos.extend(list(range(duration_i)))
        out_attr = torch.LongTensor(out_attr).to(x.device)
        out_pos = torch.LongTensor(out_pos).to(x.device)
        z = self.out_attr_emb(out_attr) + self.out_pos_emb(out_pos)
        z = z.expand(x.shape[0], -1, -1)
        print(z.shape, durations)

        # Encode
        z = self.enc_layers(
            tgt=z,
            memory=x,
            memory_key_padding_mask=mask)
        print(z.shape)

        # Output layer
        z = self.out_layer(z)
        print(z.shape)

        return z, durations
