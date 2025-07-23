#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# Configuration, including dimension settings, algorithm parameter settings.
# The last few configurations in the file are for the Kaiwu platform to use and should not be changed.
# 配置，包含维度设置，算法参数设置，文件的最后一些配置是开悟平台使用不要改动
class Config:

    # Discount factor GAMMA in RL
    # RL中的回报折扣GAMMA
    GAMMA = 0.99

    # tdlambda
    TDLAMBDA = 0.95

    # Initial learning rate
    # 初始的学习率
    START_LR = 0.0003

    # entropy regularization coefficient
    # 熵正则化系数
    BETA_START = 0.0005

    # clip parameter
    # 裁剪参数
    CLIP_PARAM = 0.2

    # value function loss coefficient
    # 价值函数损失的系数
    VF_COEF = 1

    # actions
    # 动作
    ACTION_LEN = 1
    ACTION_NUM = 8

    # features
    # 特征
    FEATURES = [
        2,
        6,
        6,
        8,
    ]

    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    VALUE_NUM = 1
    DATA_SPLIT_SHAPE = [
        FEATURE_LEN,
        VALUE_NUM,
        VALUE_NUM,
        VALUE_NUM,
        VALUE_NUM,
        ACTION_LEN,
        ACTION_LEN,
        ACTION_NUM,
    ]
    data_len = sum(DATA_SPLIT_SHAPE)

    # Input dimension of reverb sample on learner. Note that different algorithms have different dimensions.
    # **Note**, this item must be configured correctly and should be aligned with the NumpyData2SampleData function data in definition.py
    # Otherwise the sample dimension error may be reported
    # learner上reverb样本的输入维度
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = data_len
