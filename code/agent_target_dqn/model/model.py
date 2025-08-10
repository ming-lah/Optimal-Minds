#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch
import torch.nn as nn
from typing import List
from agent_target_dqn.conf.conf import Config

# -------- util --------
def make_fc_layer(in_f: int, out_f: int) -> nn.Linear:
    layer = nn.Linear(in_f, out_f)
    nn.init.orthogonal_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


# -------- 新模型主体 --------
class DuelingAttentionDQN(nn.Module):
    """
    - 按 FEATURE_SPLIT_SHAPE 将输入拆分为 N 个 token  
    - 2 层 Transformer Encoder 做全局交互  
    - Dueling Head 输出 Q 值
    """
    def __init__(self, state_shape: int, action_shape: int = 0, *, softmax: bool = False,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.obs_dim = state_shape
        self.act_dim = action_shape
        self.pool_gate = nn.Parameter(torch.tensor(0.5))

        # 1) token projection
        self.token_proj = nn.ModuleList([
            make_fc_layer(feat_len, d_model) for feat_len in Config.FEATURE_SPLIT_SHAPE
        ])
        self.seq_len = len(Config.FEATURE_SPLIT_SHAPE)

        # 2) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3) Dueling heads
        self.value_stream = nn.Sequential(
            make_fc_layer(d_model, 64), nn.ReLU(), make_fc_layer(64, 1)
        )
        self.adv_stream = nn.Sequential(
            make_fc_layer(d_model, 64), nn.ReLU(), make_fc_layer(64, action_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [batch, obs_dim]  —— 与旧版一致
        """
        # 拆分大向量 → token list
        chunks = torch.split(x, Config.FEATURE_SPLIT_SHAPE, dim=1)  # len == seq_len
        tokens = [proj(chunk.float()) for proj, chunk in zip(self.token_proj, chunks)]
        tokens = torch.stack(tokens, dim=1)  # [B, seq_len, d_model]

        # 自注意力编码
        enc = self.transformer(tokens)               # [B, seq_len, d_model]
        cls = enc[:, 0, :]
        mean = enc[:, 1:, :].mean(dim=1)
        gate = torch.sigmoid(self.pool_gate)
        feat = cls + gate * mean                          # 取第 0 个 token 做汇聚

        # Dueling
        V = self.value_stream(feat)                  # [B, 1]
        A = self.adv_stream(feat)                    # [B, act_dim]
        Q = V + A - A.mean(dim=1, keepdim=True)      # [B, act_dim]
        return Q


# 维持旧引用不变
Model = DuelingAttentionDQN