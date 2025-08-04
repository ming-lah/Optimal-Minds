#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
from kaiwu_agent.utils.common_func import attached, create_cls
from agent_target_dqn.conf.conf import Config
import math

# The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# and the remaining parameters are the attributes of the class, which should have a default value of None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    move_dir=None,
    use_talent=None,
)


SampleData = create_cls(
    "SampleData",
    obs=None,
    _obs=None,
    obs_legal=None,
    _obs_legal=None,
    act=None,
    rew=None,
    ret=None,
    done=None,
)

RelativeDistance = {
    "RELATIVE_DISTANCE_NONE": 0,
    "VerySmall": 1,
    "Small": 2,
    "Medium": 3,
    "Large": 4,
    "VeryLarge": 5,
}


RelativeDirection = {
    "East": 1,
    "NorthEast": 2,
    "North": 3,
    "NorthWest": 4,
    "West": 5,
    "SouthWest": 6,
    "South": 7,
    "SouthEast": 8,
}

DirectionAngles = {
    1: 0,
    2: 45,
    3: 90,
    4: 135,
    5: 180,
    6: 225,
    7: 270,
    8: 315,
}


# def reward_process(end_dist, history_dist):
#     # step reward
#     # 步数奖励
#     step_reward = -0.001

#     # end reward
#     # 终点奖励
#     end_reward = -0.02 * end_dist

#     # distance reward
#     # 距离奖励
#     dist_reward = min(0.001, 0.05 * history_dist)

#     return [step_reward + dist_reward + end_reward]

def reward_process(
    end_dist, 
    history_dist,
    visit_penalty: float = 0.0,
    turn_angle: float = 0.0, 
    flash_used=False, 
    d_before=None, 
    d_after=None, 
    stuck_penalty: float = 0.0,
    obs = None,
    extra_info = None,
    epoch = None
    ):

    # 获取宝箱信息
    total_chests = extra_info["game_info"]["treasure_count"]
    collected_chests = obs["score_info"]["treasure_collected_count"]
    remaining_chests = total_chests - collected_chests
    all_chest_collected = (remaining_chests == 0)
    nearest_chest_dist = _get_nearest_chest_distance(obs, extra_info)

    # step reward
    # 步数奖励
    step_reward = -0.002

    # distance reward
    # 距离奖励
    dist_reward = min(0.005, 0.20 * history_dist)

    # 终点附近奖励
    cone_reward = 0.3 * (0.3 - end_dist) if end_dist < 0.3 else 0.0

    # 访问重复惩罚
    repeat_penalty = visit_penalty

    # 转向惩罚/直线奖励
    turn_penalty = -0.002 * (turn_angle / 90.)
    straight_bonus = 0.002 if turn_angle == 0 else 0.0

    # 闪现奖励
    flash_cost = -0.02 if flash_used else 0.0
    flash_gain = 0.0
    flash_fail = 0.0
    if flash_used and d_before is not None and d_after is not None:
        flash_gain = 0.1 * max(0.0, (d_before - d_after))
        flash_fail = -0.05 if d_after >= d_before else 0.0

    # 判断训练阶段
    if epoch < 300:
        stage = 1   # 学会行走
    elif 300 <= epoch < 1000:
        stage = 2   # 尝试收集宝箱
    else:
        stage = 3   # 先找宝箱，再找终点

    # 阶段化训练
    chest_reward = 0.0
    chest_proximity_reward = 0.0
    end_success = 0.0
    end_dist_penalty = 0.0

    if stage == 1:
        if end_dist < 1e-3:
            end_success = 1.0

        end_dist_penalty = 0.01 * end_dist  # 稍微鼓励接近终点

    elif stage == 2:
        chest_reward = 0.5 * collected_chests
        chest_proximity_reward = 0.3 * (1 - nearest_chest_dist)

        if end_dist < 1e-3:
            end_success = 0.3

        end_dist_penalty = 0.02 * end_dist

    elif stage == 3:
        chest_reward = 0.5 * collected_chests
        chest_proximity_reward = 0.4 * (1 - nearest_chest_dist)

        if end_dist < 1e-3 and all_chests_collected:
            end_success = 1.0
        elif end_dist < 1e-3 and not all_chests_collected:
            end_success = -0.5  # 惩罚未收集完就去终点

        end_dist_penalty = 0.03 * end_dist

    base_reward = (step_reward + dist_reward + cone_reward + repeat_penalty + 
            turn_penalty + straight_bonus + flash_cost + flash_gain + 
            flash_fail + stuck_penalty + end_success)

    total_reward = base_reward + chest_reward + chest_proximity_reward - end_dist_penalty

    return [np.clip(total_reward, -1.5, 1.5)]

# 计算到最近未收集宝箱的归一化距离
def _get_nearest_chest_distance(obs, extra_info):
    organs = obs["frame_state"]["organs"]
    chests = [
        (o["pos"]["x"], o["pos"]["z"])
        for o in organs 
        if o["sub_type"] == 1 and o["status"] == 1  # 类型1且未收集
    ]
    if not chests:
        return 1.0  # 无宝箱时返回最大值
    
    cur_pos = (extra_info["game_info"]["pos"]["x"], 
               extra_info["game_info"]["pos"]["z"])
    max_map_dist = math.hypot(128, 128)  # 地图对角线距离
    distances = [math.hypot(p[0]-cur_pos[0], p[1]-cur_pos[1]) for p in chests]

    return min(distances) / max_map_dist  # 归一化


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


@attached
def SampleData2NumpyData(g_data):
    return np.hstack(
        (
            np.array(g_data.obs, dtype=np.float32),
            np.array(g_data._obs, dtype=np.float32),
            np.array(g_data.obs_legal, dtype=np.float32),
            np.array(g_data._obs_legal, dtype=np.float32),
            np.array(g_data.act, dtype=np.float32),
            np.array(g_data.rew, dtype=np.float32),
            np.array(g_data.ret, dtype=np.float32),
            np.array(g_data.done, dtype=np.float32),
        )
    )


@attached
def NumpyData2SampleData(s_data):
    obs_data_size = Config.DIM_OF_OBSERVATION
    legal_data_size = Config.DIM_OF_ACTION
    return SampleData(
        obs=s_data[:obs_data_size],
        _obs=s_data[obs_data_size : 2 * obs_data_size],
        obs_legal=s_data[2 * obs_data_size : 2 * obs_data_size + legal_data_size],
        _obs_legal=s_data[2 * obs_data_size + legal_data_size : 2 * obs_data_size + 2 * legal_data_size],
        act=s_data[-4],
        rew=s_data[-3],
        ret=s_data[-2],
        done=s_data[-1],
    )
