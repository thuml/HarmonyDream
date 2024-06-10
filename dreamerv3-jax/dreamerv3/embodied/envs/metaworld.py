import functools
import os

import embodied
import gym
import numpy as np


class MetaWorld(embodied.Env):

    def __init__(self, name, seed=None, repeat=1, size=(64, 64), camera="corner", use_gripper=False):
        import metaworld
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
        )

        os.environ["MUJOCO_GL"] = "egl"
        name = "-".join(name.split("_"))
        task = f"{name}-v2-goal-observable"
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed)
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = repeat
        self._use_gripper = use_gripper
        self._done = True

        self._camera = camera

    @functools.cached_property
    def obs_space(self):
        spaces = {
            "image": embodied.Space(np.uint8, self._size + (3,), 0, 255),
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
            "success": embodied.Space(bool),
        }
        if self._use_gripper:
            spaces["gripper_image"] = embodied.Space(np.uint8, self._size + (3,), 0, 255)
        return spaces

    @functools.cached_property
    def act_space(self):
        spec = self._env.action_space
        spec = spec if isinstance(spec, dict) else {'action': spec}
        return {
            'reset': embodied.Space(bool),
            **{k or 'action': self._convert(v) for k, v in spec.items()},
        }

    def step(self, action):
        action = action.copy()
        reset = action.pop('reset')
        if reset or self._done:
            self._done = False
            return self.reset()
        assert np.isfinite(action["action"]).all(), action["action"]
        reward = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(action["action"])
            success = float(info["success"])
            reward += rew or 0.0
            if done:
                self._done = True
            if done or success == 1.0:
                break
        assert success in [0.0, 1.0]
        obs = {
            "reward": np.float32(reward),
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ),
            "success": success,
        }
        if self._use_gripper:
            obs["gripper_image"] = self._env.sim.render(
                *self._size, mode="offscreen", camera_name="behindGripper"
            )
        return obs

    def reset(self):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        state = self._env.reset()[0]
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ),
            #"state": state,
            "success": 0.0,
        }
        if self._use_gripper:
            obs["gripper_image"] = self._env.sim.render(
                *self._size, mode="offscreen", camera_name="behindGripper"
            )
        return obs

    def _convert(self, space):
        if hasattr(space, 'num_values'):
            return embodied.Space(space.dtype, (), 0, space.num_values)
        elif hasattr(space, 'minimum'):
            assert np.isfinite(space.minimum).all(), space.minimum
            assert np.isfinite(space.maximum).all(), space.maximum
            return embodied.Space(
                space.dtype, space.shape, space.minimum, space.maximum)
        else:
            return embodied.Space(space.dtype, space.shape, None, None)