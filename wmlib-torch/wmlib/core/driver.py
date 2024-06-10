import numpy as np
import torch

import wmlib.envs as envs


class Driver:

    def __init__(self, envs, device, **kwargs):
        self._envs = envs
        self._device = device
        self._kwargs = kwargs
        self._on_steps = []
        self._on_resets = []
        self._on_episodes = []
        self._act_spaces = [env.act_space for env in envs]
        self.reset()

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_reset(self, callback):
        self._on_resets.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def reset(self):
        self._obs = [None] * len(self._envs)
        self._eps = [None] * len(self._envs)
        self._state = None

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            obs = {
                i: self._envs[i].reset()
                for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}
            for i, ob in obs.items():
                self._obs[i] = ob() if callable(ob) else ob
                act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
                tran = {k: self._convert(v) for k, v in {**self._obs[i], **act}.items()}
                [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]
                self._eps[i] = [tran]

            obs = {k: torch.from_numpy(np.stack([o[k] for o in self._obs])).float()
                   for k in self._obs[0]}  # convert before sending
            if len(obs['image'].shape) == 4:
                obs['image'] = obs['image'].permute(0, 3, 1, 2)
            from .. import ENABLE_FP16
            dtype = torch.float16 if ENABLE_FP16 else torch.float32  # only on cuda
            obs = {k: v.to(device=self._device, dtype=dtype) for k, v in obs.items()}

            actions, self._state = policy(obs, self._state, **self._kwargs)
            actions = [
                {k: np.array(actions[k][i]) for k in actions}
                for i in range(len(self._envs))]
            assert len(actions) == len(self._envs)
            obs = [e.step(a) for e, a in zip(self._envs, actions)]
            for i, (act, _ob) in enumerate(zip(actions, obs)):
                ob = _ob() if callable(_ob) else _ob
                self._obs[i] = ob
                tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
                [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
                self._eps[i].append(tran)
                step += 1
                if ob['is_last']:
                    ep = self._eps[i]
                    ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
                    [fn(ep, env_id=i, **self._kwargs) for fn in self._on_episodes]
                    episode += 1

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float32)
        elif np.issubdtype(value.dtype, np.signedinteger):
            return value.astype(np.int32)
        elif np.issubdtype(value.dtype, np.uint8):
            return value.astype(np.uint8)
        return value


class EvalDriver(Driver):

    def __init__(self, envs, device, init_reset_blocking=False, **kwargs):
        super().__init__(envs, device, **kwargs)
        self._init_reset = False
        self._init_reset_blocking = init_reset_blocking

    def __call__(self, policy, steps=0, episodes=0):
        self.reset()
        ongoing = [False] * len(self._envs)

        step, episode = 0, 0
        while step < steps or episode < episodes:
            if not any(ongoing):
                obs = {
                    i: self._envs[i].reset(blocking=(not self._init_reset and self._init_reset_blocking)) if isinstance(
                        self._envs[i], envs.Async) else self._envs[i].reset()
                    for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}
                for i, ob in obs.items():
                    self._obs[i] = ob() if callable(ob) else ob
                    act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
                    tran = {k: self._convert(v) for k, v in {**self._obs[i], **act}.items()}
                    [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]
                    self._eps[i] = [tran]
                ongoing = [True] * len(self._envs)
                self._init_reset = True

            obs = {k: torch.from_numpy(np.stack([o[k] for o in self._obs])).float()
                   for k in self._obs[0]}  # convert before sending
            if len(obs['image'].shape) == 4:
                obs['image'] = obs['image'].permute(0, 3, 1, 2)
            from .. import ENABLE_FP16
            dtype = torch.float16 if ENABLE_FP16 else torch.float32  # only on cuda
            obs = {k: v.to(device=self._device, dtype=dtype) for k, v in obs.items()}

            actions, self._state = policy(obs, self._state, **self._kwargs)
            actions = [
                {k: np.array(actions[k][i]) for k in actions}
                for i in range(len(self._envs))]
            assert len(actions) == len(self._envs)
            obs = [e.step(a) if ongoing[i] else self._obs[i] for i, (e, a) in enumerate(zip(self._envs, actions))]
            for i, (act, _ob) in enumerate(zip(actions, obs)):
                if not ongoing[i]:
                    continue
                ob = _ob() if callable(_ob) else _ob
                self._obs[i] = ob
                tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
                [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
                self._eps[i].append(tran)
                step += 1
                if ob['is_last']:
                    ep = self._eps[i]
                    ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
                    if episode < episodes:
                        [fn(ep, env_id=i, **self._kwargs) for fn in self._on_episodes]
                    episode += 1
                    ongoing[i] = False
