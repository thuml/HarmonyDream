import torch
import torch.nn as nn

from .base import BaseEncoder
from ..modules import *


class SamePadEncoderResnet(BaseEncoder):

    def __init__(self,
                 shapes,
                 cnn_keys=r".*",
                 mlp_keys=r".*",
                 act="elu",
                 norm="none",
                 cnn_depth=48,
                 mlp_layers=[400, 400, 400, 400],
                 blocks=0,
                 resize="stride",
                 minres=4,
                 **dummy_kwargs
                 ):
        super().__init__(shapes, cnn_keys, mlp_keys, mlp_layers)

        self._act = get_act(act)
        self._cnn_depth = cnn_depth
        self._norm = norm

        self._blocks = blocks
        self._resize = resize
        self._minres = minres

    def _cnn(self, data):
        x = torch.cat(list(data.values()), -1)
        x = x.to(memory_format=torch.channels_last)

        # x is in shape (batch, channels, height, width)
        stages = int(np.log2(x.shape[-2]) - np.log2(self._minres))
        depth = self._cnn_depth
        # print(x.shape)
        for i in range(stages):
            # preact = False
            if self._resize == 'stride':
                x = self.get(f'conv{i}', Conv2dSame, x.shape[1], depth, 4, 2)(x)
                x = self.get(f'convnorm{i}', NormLayer, self._norm, x.shape[-3:])(x)
                x = self._act(x)
            else:
                raise NotImplementedError(self._resize)
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
                depth *= 2
        if self._blocks:
            x = get_act(self._kw['act'])(x)
        return x.reshape(tuple(x.shape[:-3]) + (-1,))
