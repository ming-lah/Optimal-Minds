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

    # features
    # 特征
    FEATURES = [
        2,
        14,
        14,
        14,
        14,
        8 + 8,
        6,
        8,
        8,
    ]

    FEATURE_SPLIT_SHAPE = FEATURES

    # Size of observation
    # observation的维度
    DIM_OF_OBSERVATION = sum(FEATURES)

    # Dimension of movement action direction
    # 移动动作方向的维度
    DIM_OF_ACTION_DIRECTION = 8

    # Dimension of flash action direction
    # 闪现动作方向的维度
    DIM_OF_TALENT = 8

    # new_action
    DIM_OF_ACTION = 16
    
    # Input dimension of reverb sample on learner. Note that different algorithms have different dimensions.
    # **Note**, this item must be configured correctly and should be aligned with the NumpyData2SampleData function data in definition.py
    # Otherwise the sample dimension error may be reported
    # learner上reverb样本的输入维度
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = 2 * (DIM_OF_OBSERVATION + DIM_OF_ACTION) + 4

    # Update frequency of target network
    # target网络的更新频率
    TARGET_UPDATE_FREQ = 500

    # Discount factor GAMMA in RL
    # RL中的回报折扣GAMMA
    GAMMA = 0.97

    # epsilon
    EPSILON_MIN = 0.05
    EPSILON_MAX = 1.0
    EPSILON_DECAY = 1e-6

    # Initial learning rate
    # 初始的学习率
    START_LR = 1e-4

    # 训练调度
    S1_STEPS = 1000
    S2_STEPS = 3000

    # 宝箱
    TOTAL_TREASURES = 8

    # 潜势函数
    BASE_W_T = 0.5
    BASE_W_E = 1.0

    TREASURE_REWARD = 1.0

    GOAL_REWARD = 3.0
    INCOMPLETE_END_PENALTY = 2.0

    REWARD_CLIP = 5.0

