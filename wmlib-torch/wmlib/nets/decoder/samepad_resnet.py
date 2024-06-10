import torch
import torch.nn as nn

from .base import BaseDecoder
from ..modules import *


class SamePadDecoderResnet(BaseDecoder):
    def __init__(self, shapes, cnn_keys=r".*", mlp_keys=r".*", act="elu", norm="none", cnn_depth=48,
                 mlp_layers=[400, 400, 400, 400], blocks=0, resize="stride", minres=4, cnn_dist="mse", mlp_dist="symlog_mse", **dummy_kwargs):
        super().__init__(shapes, cnn_keys, mlp_keys, mlp_layers, cnn_dist, mlp_dist)

        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._blocks = blocks
        self._resize = resize
        self._minres = minres

    def _cnn(self, features):
        channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
        # self._shapes["image"] is in shape (height, width, channels)
        stages = int(np.log2(self._shapes['image'][-3]) - np.log2(self._minres))
        depth = self._cnn_depth * 2 ** (stages - 1)
        # TODO: this is a hack to make the decoder work with the image encoder
        if self._shapes["image"] == (160, 256, 3):
            x = self.get("convin", nn.Linear, features.shape[-1], 48 * 5 * 8)(features)
            x = torch.reshape(x, [-1, 48, 5, 8]).to(memory_format=torch.channels_last)
        else:
            x = self.get("convin", nn.Linear, features.shape[-1], self._minres * self._minres * depth)(features)
            x = torch.reshape(x, [-1, depth, self._minres, self._minres]).to(memory_format=torch.channels_last)

        for i in range(stages):
            # currently self._blocks==0
            for j in range(self._blocks):
                skip = x
                # preact = True
                x = self.get(f'conv1norm{i}b{j}', NormLayer, self._norm, x.shape[-3:])(x)
                x = self._act(x)
                x = self.get(f'conv1{i}b{j}', Conv2dSame, x.shape[1], depth, 3)(x)

                x = self.get(f'conv2norm{i}b{j}', NormLayer, self._norm, x.shape[-3:])(x)
                x = self._act(x)
                x = self.get(f'conv2{i}b{j}', Conv2dSame, x.shape[1], depth, 3)(x)
                x += skip
            depth //= 2
            act, norm = self._act, self._norm
            # preact = False
            if i == stages - 1:
                depth, act, norm = sum(channels.values()), get_act("none"), "none"
            if self._resize == 'stride':
                x = self.get(f'conv{i}', ConvTranspose2dSame, x.shape[1], depth, 4, 2)(x)
                x = self.get(f'convnorm{i}', NormLayer, norm, x.shape[-3:])(x)
                x = act(x)
            else:
                raise NotImplementedError(self._resize)

        x = x.reshape(features.shape[:-1] + x.shape[1:])
        means = torch.split(x, list(channels.values()), 2)
        dists = {
            key: core.dists.Independent(core.dists.MSE(mean), 3)
            for (key, shape), mean in zip(channels.items(), means)
        }
        return dists
