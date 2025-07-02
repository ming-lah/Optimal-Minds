#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
from kaiwu_agent.utils.common_func import create_cls, attached


SampleData = create_cls("SampleData", state=None, action=None, reward=None, next_state=None)


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


def reward_shaping(frame_no, score, terminated, truncated, obs, _obs):
    reward = 0
    end_treasure_dists = obs["feature"]
    _end_treasure_dists = _obs["feature"]

    # The reward for winning
    # 奖励1. 获胜的奖励
    if terminated:
        reward += score

    # # The reward for being close to the finish line
    # # 奖励2. 靠近终点的奖励:
    # end_dist, _end_dist = end_treasure_dists[0], _end_treasure_dists[0]
    # if end_dist > _end_dist:
    #     reward += 0.1

    # The reward for obtaining a treasure chest
    # 奖励3. 获得宝箱的奖励
    if score > 0 and not terminated:
        reward += score

    # # The reward for being close to the treasure chest (considering only the nearest one)
    # # 奖励4. 靠近宝箱的奖励(只考虑最近的那个宝箱)
    # treasure_dist, _treasure_dist = end_treasure_dists[1:], _end_treasure_dists[1:]
    # nearest_treasure_index = np.argmin(treasure_dist)
    # if treasure_dist[nearest_treasure_index] > _treasure_dist[nearest_treasure_index]:
    #     reward += 0.2

    return reward
