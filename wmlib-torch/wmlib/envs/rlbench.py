import gym
import numpy as np


class RLBench:
    def __init__(
        self,
        name,
        size=(64, 64),
        action_repeat=1,
    ):
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointPosition
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig
        from rlbench.tasks import ReachTarget, SlideBlockToTarget, TakeLidOffSaucepan
        from .rb.push_button import PushButton
        from functools import partial
        try:
            from pyrep.errors import ConfigurationPathError, IKError
            from rlbench.backend.exceptions import InvalidActionError
        except:
            pass

        # we only support reach_target in this codebase
        obs_config = ObservationConfig()
        obs_config.left_shoulder_camera.set_all(False)
        obs_config.right_shoulder_camera.set_all(False)
        obs_config.overhead_camera.set_all(False)
        obs_config.wrist_camera.set_all(False)
        obs_config.front_camera.image_size = size
        obs_config.front_camera.depth = False
        obs_config.front_camera.point_cloud = False
        obs_config.front_camera.mask = False

        action_mode = partial(JointPosition, absolute_mode=False)

        env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=action_mode(), gripper_action_mode=Discrete()
            ),
            obs_config=obs_config,
            headless=True,
            shaped_rewards=True,
        )
        env.launch()

        if name == "reach_target":
            task = ReachTarget
        elif name == "push_button":
            task = PushButton
        else:
            raise ValueError(name)
        self._env = env
        self._task = env.get_task(task)

        _, obs = self._task.reset()
        self._prev_obs = None

        self._size = size
        self._action_repeat = action_repeat

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        action = gym.spaces.Box(
            low=-1.0, high=1.0, shape=self._env.action_shape, dtype=np.float32
        )
        return {"action": action}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        try:
            reward = 0.0
            for i in range(self._action_repeat):
                obs, reward_, terminal = self._task.step(action["action"])
                success, _ = self._task._task.success()
                reward += reward_
                if terminal:
                    break
            self._prev_obs = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            success = False
            reward = 0.0
            obs = self._prev_obs

        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": terminal,
            "is_terminal": terminal,
            "image": obs.front_rgb,
            "success": success,
        }
        return obs

    def reset(self):
        _, obs = self._task.reset()
        self._prev_obs = obs
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": obs.front_rgb,
            "success": False,
        }
        return obs
