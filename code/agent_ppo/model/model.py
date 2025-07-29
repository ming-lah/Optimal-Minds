#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from typing import List
import torch
from torch import nn
import numpy as np
from agent_ppo.conf.conf import Config

import sys
import os

if os.path.basename(sys.argv[0]) == "learner.py":
    import torch

    torch.set_num_interop_threads(2)
    torch.set_num_threads(2)
else:
    import torch

    torch.set_num_interop_threads(4)
    torch.set_num_threads(4)


class NetworkModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        # feature configure parameter
        # 特征配置参数
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.feature_split_shape = Config.FEATURE_SPLIT_SHAPE
        self.label_size = Config.ACTION_NUM
        self.feature_len = Config.FEATURE_LEN
        self.value_num = Config.VALUE_NUM

        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF

        self.clip_param = Config.CLIP_PARAM

        self.data_len = Config.data_len

        # Main MLP network
        # 主MLP网络
        self.main_fc_dim_list = [self.feature_len, 128, 256]
        self.main_mlp_net = MLP(self.main_fc_dim_list, "main_mlp_net", non_linearity_last=True)
        self.label_mlp = MLP([256, 64, self.label_size], "label_mlp")
        self.value_mlp = MLP([256, 64, self.value_num], "value_mlp")

    def process_legal_action(self, label, legal_action):
        label_max, _ = torch.max(label * legal_action, 1, True)
        label = label - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return label

    def forward(self, feature, legal_action):
        # Main MLP processing
        # 主MLP处理
        fc_mlp_out = self.main_mlp_net(feature)

        # Action and value processing
        # 处理动作和值
        label_mlp_out = self.label_mlp(fc_mlp_out)
        label_out = self.process_legal_action(label_mlp_out, legal_action)

        prob = torch.nn.functional.softmax(label_out, dim=1)
        value = self.value_mlp(fc_mlp_out)

        return prob, value


class NetworkModelActor(NetworkModelBase):
    def format_data(self, obs, legal_action):
        return (
            torch.tensor(obs).to(torch.float32),
            torch.tensor(legal_action).to(torch.float32),
        )


class NetworkModelLearner(NetworkModelBase):
    def format_data(self, datas):
        return datas.view(-1, self.data_len).float().split(self.data_split_shape, dim=1)

    def forward(self, data_list, inference=False):
        feature = data_list[0]
        legal_action = data_list[-1]
        return super().forward(feature, legal_action)


def make_fc_layer(in_features: int, out_features: int):
    # Wrapper function to create and initialize a linear layer
    # 创建并初始化一个线性层
    fc_layer = nn.Linear(in_features, out_features)

    # initialize weight and bias
    # 初始化权重及偏移量
    nn.init.orthogonal(fc_layer.weight)
    nn.init.zeros_(fc_layer.bias)

    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        # Create a MLP object
        # 创建一个 MLP 对象
        super().__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            # no relu for the last fc layer of the mlp unless required
            # 除非有需要，否则 mlp 的最后一个 fc 层不使用 relu
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
