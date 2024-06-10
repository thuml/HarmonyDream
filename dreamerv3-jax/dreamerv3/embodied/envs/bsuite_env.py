import functools
import os

import embodied
import numpy as np
import bsuite

class BSuite(embodied.Env):

  def __init__(self, env, repeat=1, render=True, size=(64, 64), camera=-1):
    if isinstance(env, str):
        self._env = bsuite.load_from_id(env)
    from . import from_dm
    self._env = from_dm.FromDM(self._env)
    self._env = embodied.wrappers.ExpandScalars(self._env)
    self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
    self._size = size

  @functools.cached_property
  def obs_space(self):
    spaces = self._env.obs_space.copy()
    spaces['observation'] = embodied.Space(np.float32, (32,32,1))
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    obs = self._env.step(action)
    observation = obs['observation']
    observation = observation.reshape((28,28,1))
    # pad obs into 32x32x1
    observation = np.pad(observation, ((2,2),(2,2),(0,0)), 'constant', constant_values=0)
    obs['observation']=observation
    return obs
