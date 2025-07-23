#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import numpy as np
import torch
import os
import time
from agent_ppo.model.model import NetworkModelLearner
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, device, logger, monitor):
        self.device = device
        self.model = NetworkModelLearner().to(self.device)
        self.lr = Config.START_LR
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.logger = logger
        self.monitor = monitor
        self.last_report_monitor_time = 0
        self.label_size = Config.ACTION_NUM
        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM

    def learn(self, list_sample_data):
        results = {}
        self.model.train()
        self.optimizer.zero_grad()

        list_npdata = [torch.as_tensor(sample_data.npdata, device=self.device) for sample_data in list_sample_data]
        _input_datas = torch.stack(list_npdata, dim=0)
        data_list = self.model.format_data(_input_datas)
        rst_list = self.model(data_list)
        total_loss, info_list = self.compute_loss(data_list, rst_list)
        results["total_loss"] = total_loss.item()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters, 0.5)
        self.optimizer.step()

        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info = [i.detach().cpu().item() if torch.is_tensor(i) else i for i in info]
            else:
                _info = info.detach().mean().cpu().item() if torch.is_tensor(info) else info
            _info_list.append(_info)

        if self.monitor:
            now = time.time()
            if now - self.last_report_monitor_time >= 60:
                results["value_loss"] = round(_info_list[1], 2)
                results["policy_loss"] = round(_info_list[2], 2)
                results["entropy_loss"] = round(_info_list[3], 2)
                results["reward"] = _info_list[-1]
                self.monitor.put_data({os.getpid(): results})
                self.last_report_monitor_time = now

    def compute_loss(self, data_list, rst_list):
        (
            feature,
            reward,
            old_value,
            tdret,
            adv,
            old_action,
            old_prob,
            legal_action,
        ) = data_list

        # value loss
        # 价值损失
        value = rst_list[1].squeeze(1)
        old_value = old_value
        adv = adv
        tdret = tdret
        value_clip = old_value + (value - old_value).clamp(-self.clip_param, self.clip_param)
        value_loss1 = torch.square(tdret - value_clip)
        value_loss2 = torch.square(tdret - value)
        value_loss = 0.5 * torch.maximum(value_loss1, value_loss2).mean()

        # entropy loss
        # 熵损失
        prob = rst_list[0]
        entropy_loss = (-prob * torch.log(prob.clamp(1e-9, 1))).sum(1).mean()

        # policy loss
        # 策略损失
        clip_fracs = []
        one_hot_action = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size)
        new_prob = (one_hot_action * prob).sum(1, keepdim=True)
        ratio = new_prob / old_prob
        clip_fracs.append((ratio - 1.0).abs().gt(self.clip_param).float().mean())
        policy_loss1 = -ratio * adv
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        total_loss = value_loss * self.vf_coef + policy_loss - self.var_beta * entropy_loss
        info_list = [tdret.mean(), value_loss, policy_loss, entropy_loss] + clip_fracs
        info_list += [adv.mean(), adv.std(), reward.mean()]
        return total_loss, info_list
