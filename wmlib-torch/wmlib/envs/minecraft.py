import os
import time
import gc

import cv2
import gym
import numpy as np

from .minedojo_utils import *


def restore_action(action, act_space):
    if hasattr(act_space, 'nvec') and len(act_space.nvec) == 5 and (act_space.nvec == (3, 3, 4, 5, 3)).all():
        assert len(action) == 5
        output_action = [action[0], action[1], action[2], 12, 12, 0, 0, 0]

        # no_op, use, attack
        act_use = action[4]
        if act_use == 2:
            act_use = 3
        output_action[5] = act_use

        # no_op, 2 pitch, 2 yaw
        act_cam = action[3]
        if act_cam == 1:
            output_action[3] = 11
        elif act_cam == 2:
            output_action[3] = 13
        elif act_cam == 3:
            output_action[4] = 11
        elif act_cam == 4:
            output_action[4] = 13
        return output_action
    else:
        raise NotImplementedError


class Minecraft:
    def __init__(self, name, seed=None, action_repeat=1, size=(64, 64), sim_size=(64, 64), hard_reset_every=-1):
        os.environ["MINEDOJO_HEADLESS"] = "1"
        self._name = name
        self._seed = seed
        self._size = size
        self._sim_size = sim_size
        self._action_repeat = action_repeat
        self._hard_reset_every = hard_reset_every
        self._reset_counter = 0
        self._random = np.random.RandomState(seed)
        self._init_reset = False
        self._create_env()

    def _create_env(self):
        if self._name == "hunt_cow":
            self._env = HuntCowDenseRewardEnv(attack_reward=5, success_reward=200, nav_reward_scale=1, step_penalty=0,
                                              seed=self._random.randint(1_000_000), size=self._sim_size)
        else:
            raise NotImplementedError
        self._reset_counter = 0

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "success": gym.spaces.Box(0, 1, (), dtype=bool)
        }
        return spaces

    @property
    def act_space(self):
        # [3, 3, 4, 5, 3] actions
        # fewer camera control actions (15deg step)
        # refer to the STG-Transformer code
        return {'action': gym.spaces.MultiDiscrete([3, 3, 4, 5, 3])}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        real_action = restore_action(action["action"], self.act_space["action"])
        reward = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(real_action)
            success = info["success"]
            reward += rew or 0.0
            if done:
                break
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": done,
            "is_terminal": info["is_terminal"],
            "image": cv2.resize(state['rgb'].transpose(1, 2, 0), self._size, interpolation=cv2.INTER_LINEAR) if self._size != self._sim_size else state['rgb'].transpose(1, 2, 0),
            "success": success,
        }
        return obs

    def reset(self):
        if 0 < self._hard_reset_every <= self._reset_counter:
            # a hack to hard reset the server, be careful
            self._env.env.env.env.env.env.env._server_start = False
            self._env.env.env.env.env.env.env._info_prev_reset = None
            self._env.unwrapped._sim_spec._world_generator_handlers[0].world_seed = self._random.randint(1_000_000)
            self._reset_counter = 0
            print("[Minecraft] Hard reset the server")

        while True:
            try:
                state = self._env.reset()
                if not self._init_reset:
                    self._init_reset = True
                    print("[Minecraft] First reset done")
                break
            except Exception as e:
                print(e)
                t = np.random.randint(60)
                print(f"[Minecraft] Reset failed, sleep {t} secs, retrying...")
                self._env.close()
                del self._env
                gc.collect()
                # randomly sleep for one minute
                time.sleep(t)
                self._create_env()

        self._reset_counter += 1
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": cv2.resize(state['rgb'].transpose(1, 2, 0), self._size, interpolation=cv2.INTER_LINEAR) if self._size != self._sim_size else state['rgb'].transpose(1, 2, 0),
            "success": False,
        }
        return obs
