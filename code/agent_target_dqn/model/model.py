# #!/usr/bin/env python3
# # -*- coding: UTF-8 -*-
# ###########################################################################
# # Copyright © 1998 - 2025 Tencent. All Rights Reserved.
# ###########################################################################
# """
# Author: Tencent AI Arena Authors
# """


# import torch
# import numpy as np
# from torch import nn
# import torch.nn.functional as F
# from typing import List
# from agent_target_dqn.conf.conf import Config

# import sys
# import os

# if os.path.basename(sys.argv[0]) == "learner.py":
#     import torch

#     torch.set_num_interop_threads(2)
#     torch.set_num_threads(2)
# else:
#     import torch

#     torch.set_num_interop_threads(4)
#     torch.set_num_threads(4)


# class Model(nn.Module):
#     def __init__(self, state_shape, action_shape=0, softmax=False):
#         super().__init__()
#         # feature configure parameter
#         # 特征配置参数
#         self.feature_len = Config.DIM_OF_OBSERVATION

#         # Q network
#         # Q 网络
#         self.q_mlp = MLP([self.feature_len, 256, 128, action_shape], "q_mlp")

#     # Forward inference
#     # 前向推理
#     def forward(self, feature):
#         # Action and value processing
#         logits = self.q_mlp(feature)
#         return logits


# def make_fc_layer(in_features: int, out_features: int):
#     # Wrapper function to create and initialize a linear layer
#     # 创建并初始化一个线性层
#     fc_layer = nn.Linear(in_features, out_features)

#     # initialize weight and bias
#     # 初始化权重及偏移量
#     nn.init.orthogonal(fc_layer.weight)
#     nn.init.zeros_(fc_layer.bias)

#     return fc_layer


# class MLP(nn.Module):
#     def __init__(
#         self,
#         fc_feat_dim_list: List[int],
#         name: str,
#         non_linearity: nn.Module = nn.ReLU,
#         non_linearity_last: bool = False,
#     ):
#         # Create a MLP object
#         # 创建一个 MLP 对象
#         super().__init__()
#         self.fc_layers = nn.Sequential()
#         for i in range(len(fc_feat_dim_list) - 1):
#             fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
#             self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
#             # no relu for the last fc layer of the mlp unless required
#             # 除非有需要，否则 mlp 的最后一个 fc 层不使用 relu
#             if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
#                 self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

#     def forward(self, data):
#         return self.fc_layers(data)


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
import numpy as np
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
        feat = enc[:, 0, :]                          # 取第 0 个 token 做汇聚

        # Dueling
        V = self.value_stream(feat)                  # [B, 1]
        A = self.adv_stream(feat)                    # [B, act_dim]
        Q = V + A - A.mean(dim=1, keepdim=True)      # [B, act_dim]
        return Q


# 维持旧引用不变
Model = DuelingAttentionDQN
