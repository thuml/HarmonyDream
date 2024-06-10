# Code from: https://github.com/MineDojo/MineCLIP/blob/e6c06a0245fac63dceb38bc9bd4fecd033dae735/mineclip/dense_reward/animal_zoo/hunt_cow.py

from __future__ import annotations


import minedojo
from minedojo.sim.inventory import InventoryItem
import numpy as np

from .dense_reward import AnimalZooDenseRewardWrapper


class HuntCowDenseRewardEnv(AnimalZooDenseRewardWrapper):
    def __init__(
        self,
        step_penalty: float | int,
        nav_reward_scale: float | int,
        attack_reward: float | int,
        success_reward: float | int,
        seed: int,
        size: tuple
    ):
        max_spawn_range = 10
        distance_to_axis = int(max_spawn_range / np.sqrt(2))
        spawn_range_low = (-distance_to_axis, 1, -distance_to_axis)
        spawn_range_high = (distance_to_axis, 1, distance_to_axis)

        env = minedojo.make(
            "Combat",
            target_names=["cow"],
            target_quantities=1,
            reward_weights={
                "cow": success_reward,
            },
            # start_position=pos,
            initial_inventory=[
                InventoryItem(slot=0, name="diamond_sword", variant=None, quantity=1)
            ],
            initial_mobs=["cow", "pig", "sheep"],
            initial_mob_spawn_range_low=spawn_range_low,
            initial_mob_spawn_range_high=spawn_range_high,
            image_size=size,
            world_seed=seed,
            specified_biome="sunflower_plains",
            fast_reset=True,
            fast_reset_random_teleport_range=None,
            use_voxel=True,
            use_lidar=True,
            lidar_rays=[
                (pitch, yaw, 9999)
                for pitch in [np.deg2rad(x) for x in np.linspace(0, 30, 3)]
                for yaw in [np.deg2rad(x) for x in np.linspace(-60, 60, 9)]
            ],
        )
        super().__init__(
            env=env,
            entity="cow",
            step_penalty=step_penalty,
            nav_reward_scale=nav_reward_scale,
            attack_reward=attack_reward,
        )

        # reset cmds, call before `env.reset()`
        self._reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]

        self._episode_len = 500
        self._elapsed_steps = 0
        self._first_reset = True

    def reset(self, **kwargs):
        self._elapsed_steps = 0

        if not self._first_reset:
            for cmd in self._reset_cmds:
                self.env.unwrapped.execute_cmd(cmd)
            self.unwrapped.set_time(6000)
            self.unwrapped.set_weather("clear")
        self._first_reset = False

        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._episode_len:
            done = True
        return obs, reward, done, info