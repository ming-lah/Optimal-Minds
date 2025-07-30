#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################

"""
Author: Tencent AI Arena Authors

"""

import numpy as np
import math
from agent_target_dqn.feature.definition import RelativeDistance, RelativeDirection, DirectionAngles, reward_process
from agent_target_dqn.conf.conf import Config
from collections import defaultdict
import itertools

def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)


class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = Config.DIM_OF_ACTION
        self.max_map_dist = math.hypot(128, 128)
        self.flash_range = 16.0

        # Anti-Stuck
        self.pos_hist_window = 8
        self.no_progress_penalty = 0.2
        self.loop_penalty = 0.15
        self._pos_history = []

        # TTl
        self.bad_moves = {}
        self.BAD_TTL_MOVE = 10
        self.BAD_TTL_FLASH = 3


        # visit
        self.visit_counter = defaultdict(int)

        # 方向
        self._dir_lookup = [
            (1, 0),  (1, 1),  (0, 1),  (-1, 1),
            (-1, 0), (-1,-1), (0,-1),  (1,-1),
        ]
        self.prev_action_dir = None


        self.reset()





    def reset(self):
        self.step_no = 0
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))
        self.end_pos = None
        self.is_end_pos_found = False
        self.history_pos = []

        self.bad_moves.clear()

        self.is_flashed = True
        
        self._pos_history.clear()

        self.visit_counter.clear()
        self.prev_action_dir = None

    def _get_pos_feature(self, found, cur_pos, target_pos):
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))
        dist = np.linalg.norm(relative_pos)
        target_pos_norm = norm(target_pos, 128, -128)

        direction_onehot = np.zeros(8, dtype=np.float32)
        if dist > 1e-4:
            theta = (math.degrees(math.atan2(relative_pos[1], relative_pos[0])) + 360) % 360
            dir_idx = int(((theta + 22.5) % 360) // 45)  # 0~7
            direction_onehot[dir_idx] = 1.0

        feature = np.concatenate(
            [
                [float(found)],              # 1
                direction_onehot,            # 8
                [
                    norm(relative_pos[0] / max(dist, 1e-4), 1, -1),  # 单位向量 x
                    norm(relative_pos[1] / max(dist, 1e-4), 1, -1),  # 单位向量 y
                ],                           # 2
                list(target_pos_norm),       # 2
                [norm(dist, self.max_map_dist)],  # 1
            ],
            axis=0,
        )
        return feature

    def pb2struct(self, frame_state, last_action):
        obs, extra_info = frame_state
        self.step_no = obs["frame_state"]["step_no"]

        hero = obs["frame_state"]["heroes"][0]
        self.flash_cd = hero["talent"]["status"]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

        if self.flash_cd == 1:
            self.is_flashed = True
        else:
            self.is_flashed = False

        # History position
        # 历史位置
        self.history_pos.append(self.cur_pos)
        if len(self.history_pos) > 10:
            self.history_pos.pop(0)

        # End position
        # 终点位置
        for organ in obs["frame_state"]["organs"]:
            if organ["sub_type"] == 4:
                end_pos_dis = RelativeDistance[organ["relative_pos"]["l2_distance"]]
                end_pos_dir = RelativeDirection[organ["relative_pos"]["direction"]]
                if organ["status"] != -1:
                    self.end_pos = (organ["pos"]["x"], organ["pos"]["z"])
                    self.is_end_pos_found = True
                # if end_pos is not found, use relative position to predict end_pos
                # 如果终点位置未找到，使用相对位置预测终点位置
                elif (not self.is_end_pos_found) and (
                    self.end_pos is None
                    or self.step_no % 100 == 0
                    or self.end_pos_dir != end_pos_dir
                    or self.end_pos_dis != end_pos_dis
                ):
                    distance = end_pos_dis * 20
                    theta = DirectionAngles[end_pos_dir]
                    delta_x = distance * math.cos(math.radians(theta))
                    delta_z = distance * math.sin(math.radians(theta))

                    self.end_pos = (
                        max(0, min(128, round(self.cur_pos[0] + delta_x))),
                        max(0, min(128, round(self.cur_pos[1] + delta_z))),
                    )

                    self.end_pos_dir = end_pos_dir
                    self.end_pos_dis = end_pos_dis

        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, 128, -128)
        self.feature_end_pos = self._get_pos_feature(self.is_end_pos_found, self.cur_pos, self.end_pos)

        # History position feature
        # 历史位置特征
        self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0])

        self.move_usable = True
        self.last_action = last_action

    def process(self, frame_state, last_action):
        self.pb2struct(frame_state, last_action)

        # 动作
        legal_action = self.get_legal_action()

        # ---------- 新增 3×3 walkable、射线距离 ---------- #
        cur_cell = tuple(map(int, self.cur_pos))
        offsets = list(itertools.product([-1, 0, 1], repeat=2))
        walkable = [1.0 if self._is_free((cur_cell[0]+dx, cur_cell[1]+dz)) else 0.0
                    for dx, dz in offsets]                   # 9 dims

        fwd_dir = (self.last_action % 8) if 0 <= self.last_action < 16 else None
        ray_feat = [self._cast_ray(self.cur_pos, self._dir_lookup[fwd_dir])
                    if fwd_dir is not None else 1.0]          # 1 dim

        base_feature = np.concatenate([
            self.cur_pos_norm,
            self.feature_end_pos,
            self.feature_history_pos,
            legal_action,
            walkable,
            ray_feat,
        ])        

        # Feature
        # base_feature = np.concatenate([self.cur_pos_norm, self.feature_end_pos, self.feature_history_pos, legal_action])

        end_dist = self.feature_end_pos[-1]
        r_flash = self.flash_range / self.max_map_dist
        flash_used = (last_action >= Config.DIM_OF_ACTION_DIRECTION)
        d_after = max(0.0, end_dist - r_flash) if flash_used else end_dist
        extra_feats = np.array([r_flash, end_dist, d_after, self.flash_cd / 100], dtype=np.float32)

        feature = np.concatenate([base_feature, extra_feats], axis=0)

        stuck_penalty = 0.0

        # ---------- 访问次数惩罚 ---------- #
        self.visit_counter[cur_cell] += 1
        visit_penalty = -0.03 * min(self.visit_counter[cur_cell], 5)

        # ---------- 转向角惩罚 ---------- #
        turn_angle = 0.0
        if self.prev_action_dir is not None and 0 <= self.last_action < 8:
            diff = abs(self.last_action - self.prev_action_dir) % 8
            diff = 8 - diff if diff > 4 else diff            # 0-4
            turn_angle = diff * 45
        if 0 <= self.last_action < 8:
            self.prev_action_dir = self.last_action
        self._pos_history.append(self.cur_pos)

        
        # 记录n步
        if len(self._pos_history) > self.pos_hist_window:
            self._pos_history.pop(0)

        # n步内没有移动
        if len(self._pos_history) == self.pos_hist_window and len(set(self._pos_history)) == 1:
            stuck_penalty -= self.no_progress_penalty 

        # 最近4步循环
        if len(self._pos_history) >= 4:
            if (self._pos_history[-1] == self._pos_history[-3] and 
                self._pos_history[-2] == self._pos_history[-4]):
                stuck_penalty -= self.loop_penalty

        reward = reward_process(
            end_dist,
            self.feature_history_pos[-1],
            visit_penalty,
            turn_angle,
            flash_used,
            end_dist,
            d_after,
            stuck_penalty,
        )

        return (
            feature,
            legal_action,
            reward,
        )

    # def get_legal_action(self):
    #     # if last_action is move and current position is the same as last position, add this action to bad_move_ids
    #     # 如果上一步的动作是移动，且当前位置与上一步位置相同，则将该动作加入到bad_move_ids中

    #     legal_action = [False] * self.move_action_num

    #     if self.move_usable:
    #         legal_action[:8] =  [True] * 8
    #         legal_action[8:] = [self.is_flashed] * 8

        
    #     if (
    #         0 <= self.last_action <8
    #         and abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 0.001
    #         and abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 0.001
    #     ):
    #         self.bad_move_ids.add(self.last_action)
    #     else:
    #         self.bad_move_ids = set()

    #     for move_id in self.bad_move_ids:
    #         legal_action[move_id] = 0


    #     if not any(legal_action) and self.move_usable:
    #         self.bad_move_ids.clear()
    #         legal_action[:8] = [True] * 8
    #         legal_action[8:] = [self.is_flashed] * 8

    #     return legal_action

    def get_legal_action(self):
        """
        0–7 : 普通移动方向
        8–15: 闪现方向（需 is_flashed）
        bad_moves 为 {move_id: ttl}，ttl>0 时对应方向强制不可选
        """

        for k in list(self.bad_moves.keys()):
            self.bad_moves[k] -= 1
            if self.bad_moves[k] <= 0:
                del self.bad_moves[k]

        # ---------- 1. 基础可行性 ---------- #
        legal_action = [False] * 16
        if self.move_usable:
            legal_action[:8]  = [True] * 8                       # 移动
            legal_action[8:] = [self.is_flashed] * 8             # 闪现

        # ---------- 2. 撞墙检测 + 更新 TTL ---------- #
        pos_unchanged = (
            abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 1e-3 and
            abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 1e-3
        )
        if pos_unchanged and 0 <= self.last_action < 16:
            if self.last_action < 8:
                self.bad_moves[self.last_action] = self.BAD_TTL_MOVE
            else:
                self.bad_moves[self.last_action % 8] = self.BAD_TTL_FLASH


        # ---------- 3. 屏蔽带 TTL 的方向 ---------- #
        for move_id in self.bad_moves.keys():
            legal_action[move_id] = False

        # ---------- 4. 兜底：若全被屏蔽，解锁移动方向 ---------- #
        if self.is_end_pos_found:
            dx = self.end_pos[0] - self.cur_pos[0]
            dz = self.end_pos[1] - self.cur_pos[1]
            if max(abs(dx), abs(dz)) <= 1:
                theta = (math.degrees(math.atan2(dz, dx)) + 360) % 360
                dir_idx = int(((theta + 22.5) % 360) // 45)
                legal_action[dir_idx] = True

    # 5. 全零兜底退回
        if not any(legal_action):
            fallback = self.prev_action_dir if self.prev_action_dir is not None else 0
            legal_action[fallback] = True

        # return legal_action
        return np.asarray(legal_action, dtype=np.float32)
    



    
    def _is_free(self, pos):
        """粗略判定坐标是否在地图内且非障碍。现阶段仅做边界检查。"""
        x, z = map(int, pos)
        return 0 <= x < 128 and 0 <= z < 128   # 128×128 地图
 
    def _cast_ray(self, start, direction, max_step=20):
        """沿 direction 走，遇障碍或越界即停，返回归一化距离 0-1。"""
        x, z = start
        dx, dz = direction
        for step in range(1, max_step + 1):
            if not self._is_free((x + dx * step, z + dz * step)):
                return step / max_step
        return 1.0
