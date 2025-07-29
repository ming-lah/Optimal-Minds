#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    BaseAgent,
    reset_wrapper,
)
from agent_diy.model.model import Model
from kaiwu_agent.utils.common_func import attached
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)

    @predict_wrapper
    def predict(self, list_obs_data):
        pass

    @exploit_wrapper
    def exploit(self, list_obs_data):
        pass

    @learn_wrapper
    def learn(self, list_sample_data):
        pass

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        pass

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        pass

    def observation_process(self, obs, extra_info):
        pass

    def action_process(self, act_data):
        pass
