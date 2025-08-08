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

        # flash
        self.max_map_dist = math.hypot(128, 128)
        self.flash_range = 16.0

        # Anti-Stuck
        self.pos_hist_window = 10
        self.no_progress_penalty = 0.2
        self.loop_penalty = 0.15
        self._pos_history = []

        # TTL
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

        # 终点附近的步数
        self.near_goal_steps = 0

        # 死角逃脱奖励
        self.corner_radius = getattr(Config, "CORNER_RADIUS", 6)
        self.map_w = getattr(Config, "MAP_WIDTH", 128)
        self.map_h = getattr(Config, "MAP_HEIGHT",128)

        # 宝箱与加速
        self.prev_treasure_count = 0
        self.prev_buff_count = 0

        # 全局特征利用
        self.prev_score = 0.0
        self.prev_treasure_phi = 0.0
        self.prev_end_phi = 0.0
        self.visible_treasures = []


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
        self.near_goal_steps = 0
        self.prev_buff_count = 0
        self.prev_treasure_count = 0

        self.prev_score = 0
        self.prev_treasure_phi = 0.0
        self.prev_end_phi = 0.0
        self.visible_treasures = []

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
        obs, _ = frame_state
        self.step_no = obs["frame_state"]["step_no"]

        # 闪现cd && buff_cd
        hero = obs["frame_state"]["heroes"][0]
        self.flash_cd = hero["talent"]["status"]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])
        self.buff_active = hero["speed_up"]              # 0/1是否在加速
        self.buff_remain = hero["buff_remain_time"]      # 剩余Buff时间
        self.buff_duration = 100       # buff总时间



        # 奖励检测
        cur_treasure_count = obs["score_info"]["treasure_collected_count"]
        cur_buff_count = obs["score_info"]["buff_count"]
        self.treasure_gain = cur_treasure_count - self.prev_treasure_count
        self.buff_gain = cur_buff_count - self.prev_buff_count
        self.prev_treasure_count = cur_treasure_count
        self.prev_buff_count = cur_buff_count

        # 全局进度特征
        cur_score = obs["score_info"]["score"]
        self.score_delta = self.prev_score - cur_score
        self.prev_score = cur_score

        # 寻找宝箱
        visible_treasures = [org for org in obs["frame_state"]["organs"] 
                              if org["sub_type"] == 1 and org["status"] == 1]
        self.visible_treasures = [
            (org["pos"]["x"], org["pos"]["z"]) for org in visible_treasures
        ]
        if visible_treasures:
            # 最近宝箱
            nearest = min(visible_treasures, key=lambda o: 
                          (o["pos"]["x"]-self.cur_pos[0])**2 + (o["pos"]["z"]-self.cur_pos[1])**2)
            target = (nearest["pos"]["x"], nearest["pos"]["z"])
            self.feature_treasure = self._get_pos_feature(1, self.cur_pos, target)
            self.cur_treasure_dist = np.linalg.norm(np.array(self.cur_pos) - np.array(target))
        else:
            # 未发现宝箱
            self.feature_treasure = np.concatenate([
                [0.0], np.zeros(8), np.zeros(2), np.zeros(2), [1.0]
            ], dtype=np.float32)
            self.cur_treasure_dist = None
        # 寻找Buff
        buff_obj = next((org for org in obs["frame_state"]["organs"] 
                         if org["sub_type"] == 2 and org["status"] == 1), None)
        if buff_obj:
            bpos = (buff_obj["pos"]["x"], buff_obj["pos"]["z"])
            self.feature_buff = self._get_pos_feature(1, self.cur_pos, bpos)
        else:
            self.feature_buff = np.concatenate([
                [0.0], np.zeros(8), np.zeros(2), np.zeros(2), [1.0]
            ], dtype=np.float32)


        # _is_free准备数据
        self.local_map = np.array(
            [row["values"] for row in obs["map_info"]],
            dtype=np.int32
        )
        self.local_h, self.local_w = self.local_map.shape

        cx, cz = self.cur_pos
        half_w, half_h = self.local_w//2, self.local_h//2
        self._map_left = cx - half_w
        self._map_top  = cz - half_h
        self._vision_radius = self.local_h // 2

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

    def process(self, frame_state, last_action, global_step, done):

        obs, _ = frame_state

        self.pb2struct(frame_state, last_action)
        legal_action = self.get_legal_action()


        # ---------------------3*3 领域特征----------------------
        cur_cell = tuple(map(int, self.cur_pos))
        offsets = [(dx,dz) for dx,dz in itertools.product([-1,0,1],repeat=2) if not (dx==0 and dz==0)]
        walkable = [1.0 if self._is_free((cur_cell[0]+dx, cur_cell[1]+dz)) else 0.0
                    for dx, dz in offsets]
        # ------------------------------------------------------


        # ------------------------射限特征----------------------
        ray_feats = []
        for dir_vec in self._dir_lookup:
            ray_feats.append(self._cast_ray(self.cur_pos, dir_vec))
        # -----------------------------------------------------


        base_feature = np.concatenate([
            self.cur_pos_norm,
            self.feature_end_pos,
            self.feature_history_pos,
            self.feature_treasure,
            self.feature_buff,
            legal_action,
            walkable,
            ray_feats,
        ])        


        # -----------------闪现奖励------------------
        end_dist = self.feature_end_pos[-1]
        r_flash = self.flash_range / self.max_map_dist
        flash_used = (last_action >= Config.DIM_OF_ACTION_DIRECTION)
        d_after = max(0.0, end_dist - r_flash) if flash_used else end_dist
        # ------------------------------------------


        # ------------------buff--------------------
        buff_time_norm = (self.buff_remain / self.buff_duration) if self.buff_duration > 0 else 0.0
        # ------------------------------------------


        # extra_feats = np.array([r_flash, end_dist, d_after, self.flash_cd / 100,
        #                         float(self.buff_active), buff_time_norm], dtype=np.float32)

        
        # --------------步数宝箱分数全局信息----------------
        step_ratio = obs["score_info"]["step_no"] /  Config.MAX_STEP
        treasure_ratio = obs["score_info"]["treasure_collected_count"] / Config.TOTAL_TREASURES
        score_delta = self.score_delta
        talent_count = obs["score_info"]["talent_count"] / 100
        # ------------------------------------------------

        extra_feats = [
            r_flash,
            end_dist,
            d_after,
            self.flash_cd / 100,
            float(self.buff_active),
            buff_time_norm,
            # 全局进度
            step_ratio,
            treasure_ratio,
            score_delta,
            talent_count,
        ]
        
        
        # --------------------障碍密度-------------------
        h, w = self.local_map.shape
        center_y, center_x = h // 2, w // 2
        r = 3
        y0 = max(0, center_y - r)
        y1 = min(h, center_y + r + 1)
        x0 = max(0, center_x - r)
        x1 = min(w, center_x + r + 1)
        submap = self.local_map[y0:y1, x0:x1]
        if submap.size > 0:
            density_7x7 = float(np.mean(submap != 0))
        else:
            density_7x7 = 0.0
        extra_feats.append(density_7x7)
        # -----------------------------------------------


        # -----------------多宝箱–终点联合特征----------------
        cx, cz = self.cur_pos
        t2e_feats = []
        # 距离排序
        vt = sorted(self.visible_treasures,
                    key=lambda p: (p[0]-cx)**2+(p[1]-cz)**2)[:3]
        for bx, bz in vt:
            dx1, dz1 = bx-cx, bz-cz
            dx2, dz2 = self.end_pos[0]-bx, self.end_pos[1]-bz if self.end_pos else (0,0)
            d1_norm = math.hypot(dx1, dz1)/self.max_map_dist
            d2_norm = math.hypot(dx2, dz2)/self.max_map_dist
            cos_ang = (dx1*dx2 + dz1*dz2) / (math.hypot(dx1,dz1)*math.hypot(dx2,dz2)+1e-6)
            t2e_feats += [d1_norm, d2_norm, cos_ang]
        while len(t2e_feats)<9: t2e_feats.append(0.0)
        # --------------------------------------------------


        feature = np.concatenate([
            base_feature,
            np.array(extra_feats, dtype=np.float32),
            np.array(t2e_feats, dtype=np.float32),
        ], axis=0)
        


        # ---------------访问次数惩罚--------------- 
        self.visit_counter[cur_cell] += 1
        visit_penalty = -0.03 * min(self.visit_counter[cur_cell], 5)
        # -------------------------------------------


        # -----------------转向角惩罚---------------
        turn_angle = 0.0
        if self.prev_action_dir is not None and 0 <= self.last_action < 8:
            diff = abs(self.last_action - self.prev_action_dir) % 8
            diff = 8 - diff if diff > 4 else diff            # 0-4
            turn_angle = diff * 45
        if 0 <= self.last_action < 8:
            self.prev_action_dir = self.last_action
        self._pos_history.append(self.cur_pos)
        # -----------------------------------------


        # --------------- Anti-Stuck---------------
        stuck_penalty = 0.0

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
        # ------------------------------------------


        # ---------------终点附近徘徊惩罚---------------
        NEAR_GOAL_TH = 0.05
        MAX_LOITER    = 10
        if end_dist < NEAR_GOAL_TH:
            self.near_goal_steps += 1
        else:
            self.near_goal_steps = 0
        near_goal_penalty = -0.05 * min(self.near_goal_steps, MAX_LOITER)
        # ---------------------------------------------


        # -----------------死角逃脱奖励----------------
        if last_action < 8:
            dx, dz = self._dir_lookup[last_action]
            next_pos = (int(self.cur_pos[0] + dx), int(self.cur_pos[1] + dz))
        else:
            dir_idx = last_action % 8
            dx, dz = self._dir_lookup[dir_idx]
            next_pos = (
                int(self.cur_pos[0] + dx * self.flash_range),
                int(self.cur_pos[1] + dz * self.flash_range)
            )
        next_pos = (
            max(0, min(self.map_w - 1, next_pos[0])),
            max(0, min(self.map_h - 1, next_pos[1])),
        )
        cur_region = self._get_window_neighbors(self.cur_pos, self.corner_radius)
        next_region = self._get_window_neighbors(next_pos,  self.corner_radius)
        v_cur  = np.mean([ self.visit_counter[c] for c in cur_region ])
        v_next = np.mean([ self.visit_counter[c] for c in next_region ])
        corner_reward = 0.05 if v_next < v_cur else 0.0
        # --------------------------------------------


        # ----------------------潜势函数 宝箱 && 终点-------------------------

        # 基本计算逻辑
        if global_step < Config.S1_STEPS:
            e, t = 1.0, 0.0
        elif global_step < Config.S2_STEPS:
            e = 1 - (global_step - Config.S1_STEPS) / (Config.S2_STEPS - Config.S1_STEPS)
        else:
            e, t= 0.0, 1.0
        gamma_0 = 0.99

        # 加大终点力度 && 拿完宝箱兜底
        e = max(e, Config.E_MIN)
        treasure_left = Config.TOTAL_TREASURES - obs["score_info"]["treasure_collected_count"]
        collected = obs["score_info"]["treasure_collected_count"]
        if treasure_left == 0:
            e = 1.0
        elif treasure_left == Config.TOTAL_TREASURES - 1:
            e = max(e, 0.6)
        t = 1 - e

        # if obs["score_info"]["treasure_collected_count"] < Config.TREASURES_BEFORE_RETURN:
        #     t, e = 1.0, 0.0
        # elif global_step < Config.S1_STEPS:
        #     frac = global_step / Config.S1_STEPS
        #     t = max(Config.T_MIN, 1.0 - frac)
        #     e = max(Config.E_MIN, frac)
        # else:
        #     t, e = Config.T_MIN, 1.0
        
        # 宝箱机制
        shape_T = 0.0
        if self.cur_treasure_dist is not None:
            cur_td_norm = self.cur_treasure_dist / self.max_map_dist
            cur_phi_t = -cur_td_norm
            prev_phi_t = getattr(self, "prev_treasure_phi", 0.0)
            shape_T     = Config.TREASURE_REWARD * (gamma_0 * cur_phi_t - prev_phi_t)
            self.prev_treasure_phi = cur_phi_t
        else:
            self.prev_treasure_phi = 0.0
        
        shape_E = 0.0
        if self.end_pos is not None:
            raw_ed      = np.linalg.norm(np.array(self.cur_pos) - np.array(self.end_pos))
            cur_ed_norm = raw_ed / self.max_map_dist
            cur_phi_e   = -cur_ed_norm
            prev_phi_e  = getattr(self, "prev_end_phi", 0.0)
            shape_E     = Config.GOAL_REWARD * (gamma_0 * cur_phi_e - prev_phi_e)
            self.prev_end_phi = cur_phi_e
        else:
            self.prev_end_phi = 0.0

        shape = t * shape_T + e * shape_E
        # -------------------------------------------------------------------


        # ------------------一次性宝箱奖励&终点奖惩-----------------------
        one_time = Config.TREASURE_IMMEDIATE_REWARD * max(0, self.treasure_gain) * t
        if collected <= Config.TOTAL_TREASURES:
            final_top3_reward = Config.FINAL_TOP3_REWARD.get(collected, 0.0) * t
        else:
            final_top3_reward = 0.0
        if done:
            # 宝箱训练中，未收集完宝箱到终点
            if t > 0.0 and obs["score_info"]["treasure_collected_count"] <= Config.TOTAL_TREASURES - 2:
                end_pen = Config.INCOMPLETE_END_PENALTY * t
            # 宝箱训练中，收集完宝箱到终点
            elif t > 0.0 and obs["score_info"]["treasure_collected_count"] == Config.TOTAL_TREASURES:
                end_pen = Config.GOAL_REWARD * e + Config.PERFECT_REWARD * t
            # 终点训练奖励
            else:
                end_pen = Config.GOAL_REWARD * e
        else:
            end_pen = 0.0  # 不能够改成负值

        # if done:
        #     if obs["score_info"]["treasure_collected_count"] < Config.TOTAL_TREASURES:
        #         end_pen = Config.INCOMPLETE_END_PENALTY * e
        #     else:
        #         end_pen = Config.GOAL_REWARD * e + Config.PERFECT_REWARD * t
        # else:
        #     end_pen = -1.0 * e
        # ----------------------------------------------------------


        # ------------------统一奖励传入--------------------
        # 将原来的终点附近徘徊惩罚以及终点附近奖励乘上系数进入
        near_goal_penalty = near_goal_penalty * e
        cone_reward = 0.3 * (0.3 - end_dist) if end_dist < 0.3 else 0.0
        cone_reward = cone_reward * e
        #--------------------------------------------------

        reward = reward_process(
            end_dist,
            self.feature_history_pos[-1],
            visit_penalty,
            turn_angle,
            flash_used,
            end_dist,
            d_after,
            stuck_penalty,
            near_goal_penalty,
            cone_reward,
            corner_reward,
            self.treasure_gain,
            self.buff_gain,
            shape,
            one_time,
            end_pen,
            final_top3_reward,
        )
        

        return (
            feature,
            legal_action,
            reward,
        )

    def get_legal_action(self):
        # TTL减一
        for k in list(self.bad_moves.keys()):
            self.bad_moves[k] -= 1
            if self.bad_moves[k] <= 0:
                del self.bad_moves[k]
        
        # 基本的设置
        legal_action = [False] * 16
        if self.move_usable:
            legal_action[:8]  = [True] * 8                       # 移动
            legal_action[8:] = [self.is_flashed] * 8             # 闪现

        # 撞墙检测+更新TTL
        pos_unchanged = (
            abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 1e-3 and
            abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 1e-3
        )
        if pos_unchanged and 0 <= self.last_action < 16:
            if self.last_action < 8:
                self.bad_moves[self.last_action] = self.BAD_TTL_MOVE
            else:
                self.bad_moves[self.last_action % 8] = self.BAD_TTL_FLASH

        # 屏蔽TTL方向
        for move_id in self.bad_moves.keys():
            legal_action[move_id] = False

        # 终点策略
        if self.is_end_pos_found:
            dx = self.end_pos[0] - self.cur_pos[0]
            dz = self.end_pos[1] - self.cur_pos[1]
            if max(abs(dx), abs(dz)) <= 1:
                theta = (math.degrees(math.atan2(dz, dx)) + 360) % 360
                dir_idx = int(((theta + 22.5) % 360) // 45)
                legal_action[dir_idx] = True

        # 全零兜底退回
        if not any(legal_action):
            fallback = self.prev_action_dir if self.prev_action_dir is not None else 0
            legal_action[fallback] = True

        # return legal_action
        return np.asarray(legal_action, dtype=np.float32)
    



    # 工具函数
    def _is_free(self, pos):
        x, z = map(int, pos)
        if not (0 <= x < 128 and 0 <= z < 128):
            return False
        lx = x - self._map_left
        lz = z - self._map_top
        if not (0 <= lx < self.local_w and 0 <= lz < self.local_h):
            return True
        return self.local_map[lz, lx] == 0
 
    def _cast_ray(self, start, direction, max_step=None):
        x0, z0 = start
        dx, dz = direction
        max_step = max_step if max_step is not None else self._vision_radius
        for step in range(1, max_step + 1):
            x, z = x0 + dx * step, z0 + dz * step
            if not self._is_free((x, z)):
                return step / max_step
        return 1.0

    def _get_window_neighbors(self, pos, radius):
        x0, y0 = pos
        cells = []
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                if dx==0 and dy==0: continue
                x, y = x0+dx, y0+dy
                if 0 <= x < self.map_w and 0 <= y < self.map_h:
                    cells.append((x,y))
        return cells
