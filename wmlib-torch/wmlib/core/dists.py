import torch
import torch.nn.functional as F
import torch.distributions as tdist


class OneHotDist(tdist.OneHotCategorical):

    def __init__(self, logits=None, probs=None, dtype=None, validate_args=False):
        self._sample_dtype = dtype or torch.float32  # FIXME currently ignored
        super().__init__(probs=probs, logits=logits, validate_args=False)  # FIXME: validate_args=None? => Error

        # FIXME event_shape -1 for now, because I think could be empty
        # if so, tf uses logits or probs shape[-1]
        self._mode = F.one_hot(torch.argmax(self.logits, -1), self.event_shape[-1]).float()  # FIXME dtype

    @property
    def mode(self):
        return self._mode

    def sample(self, sample_shape=(), seed=None):
        # Straight through biased gradient estimator.
        # FIXME seed is not possible here

        sample = super().sample(sample_shape)  # .type(self._sample_dtype) # FIXME
        probs = super().probs
        while len(probs.shape) < len(sample.shape):  # adds dims on 0
            probs = probs[None]

        sample += (probs - probs.detach())  # .type(self._sample_dtype)
        return sample

    # custom log_prob more stable
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        value, logits = torch.broadcast_tensors(value, self.logits)
        indices = value.max(-1)[1]  # FIXME can we support soft label?
        ret = -F.cross_entropy(
            logits.reshape(-1, *self.event_shape),
            indices.reshape(-1).detach(),
            reduction='none'
        )
        return torch.reshape(ret, logits.shape[:-1])


# From https://github.com/toshas/torch_truncnorm/blob/main/TruncatedNormal.py
import math
from numbers import Number

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1, ).tolist()):
            raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b -
                                 self._little_phi_a * little_phi_coeff_a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        # NOTE: additional to github.com/toshas/torch_truncnorm
        self._mode = torch.clamp(torch.zeros_like(self.a), self.a, self.b)
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def mode(self):
        return self._mode

    @property
    def variance(self):
        return self._variance

    # @property #In pytorch is a function
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        # icdf is numerically unstable; as a consequence, so is rsample.
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, scalar_a, scalar_b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, scalar_a, scalar_b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._mode = torch.clamp(self.loc, scalar_a, scalar_b)  # NOTE: additional to github.com/toshas/torch_truncnorm
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale


#
class TruncNormalDist(TruncatedNormal):

    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale, low, high)
        self._clip = clip
        self._mult = mult

        self.low = low
        self.high = high

    def sample(self, *args, **kwargs):
        event = super().rsample(*args, **kwargs)
        if self._clip:
            clipped = torch.clamp(
                event, self.low + self._clip, self.high - self._clip
            )
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


# Pytorch distributions extending with mode
class Independent(tdist.Independent):

    @property
    def mode(self):
        return self.base_dist.mode


class Normal(tdist.Normal):
    # FIXME support log_prob_without_constant
    @property
    def mode(self):
        return self.mean

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)


class MSE(tdist.Normal):

    def __init__(self, loc, validate_args=None):
        super(MSE, self).__init__(loc, 1.0, validate_args=validate_args)

    @property
    def mode(self):
        return self.mean

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # NOTE: dropped the constant term
        return -((value - self.loc) ** 2) / 2


class Bernoulli(tdist.Bernoulli):

    @property
    def mode(self):
        return torch.round(self.probs)  # >0.5

# for dreamerv3
def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class DiscDist:
    def __init__(self, logits, low=-20.0, high=20.0, transfwd=symlog, transbwd=symexp, device="cuda"):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.bins = torch.linspace(low, high, logits.shape[-1]).to(self.logits.device)
        self.transfwd = transfwd
        self.transbwd = transbwd

    @property
    def mean(self):
        _mean = self.probs * self.bins
        return self.transbwd(torch.sum(_mean, dim=-1))

    @property
    def mode(self):
        _mode = self.probs * self.bins
        return self.transbwd(torch.sum(_mode, dim=-1))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        with torch.cuda.amp.autocast(enabled=False):  # for stability
            x = x.float()
            self.logits = self.logits.float()
            self.bins = self.bins.float()

            self.bins = self.bins.to(x.device)
            x = self.transfwd(x)
            # x(time, batch, 1)
            below = (self.bins <= x[..., None]).to(torch.int32).sum(dim=-1) - 1
            above = len(self.bins) - (self.bins > x[..., None]).to(torch.int32).sum(dim=-1)
            below = torch.clip(below, 0, len(self.bins) - 1)
            above = torch.clip(above, 0, len(self.bins) - 1)
            equal = (below == above)

            dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - x))
            dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - x))
            total = dist_to_below + dist_to_above
            weight_below = dist_to_above / total
            weight_above = dist_to_below / total
            target = (
                F.one_hot(below, num_classes=len(self.bins)) * weight_below[..., None]
                + F.one_hot(above, num_classes=len(self.bins)) * weight_above[..., None]
            )
            log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
            target = target.squeeze(-2)
            return (target * log_pred).sum(-1)


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    @property
    def mode(self):
        return symexp(self._mode)

    @property
    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class MultiDiscreteDist(Distribution):

    def __init__(self, logits=None, dtype=None, validate_args=False, nvec=None):
        super().__init__(validate_args=validate_args)
        self._nvec = nvec
        self._base_dist = []
        cum_count = 0
        for i in nvec:
            self._base_dist.append(OneHotDist(
                logits=logits[..., cum_count:cum_count + i], dtype=dtype, validate_args=validate_args))
            cum_count += i
        self._mode = torch.cat([d.mode for d in self._base_dist], dim=-1)

    @property
    def mode(self):
        return self._mode

    def sample(self, sample_shape=torch.Size(), seed=None):
        return torch.cat([d.sample(sample_shape) for d in self._base_dist], dim=-1)

    def log_prob(self, value):
        cum_count = 0
        log_prob = []
        for d in self._base_dist:
            log_prob.append(d.log_prob(value[..., cum_count:cum_count + d.event_shape[-1]]))
            cum_count += d.event_shape[-1]
        log_prob = torch.stack(log_prob, dim=-1).sum(-1)
        return log_prob

    def entropy(self):
        return torch.stack([d.entropy() for d in self._base_dist], dim=-1).sum(-1)


class MWMMSEDist:
    # !From MWM, need to be merge into MSE
    def __init__(self, mode, dims, agg="sum"):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        self.batch_shape = mode.shape[: len(mode.shape) - dims]
        self.event_shape = mode.shape[len(mode.shape) - dims:]

    @property
    def mode(self):
        return self._mode

    @property
    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class MWMSymlogDist:
    # !From MWM, need to be merge into SymlogDist
    def __init__(self, mode, dims, agg="sum"):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        self.batch_shape = mode.shape[: len(mode.shape) - dims]
        self.event_shape = mode.shape[len(mode.shape) - dims:]

    @property
    def mode(self):
        return symexp(self._mode)

    @property
    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - symlog(value)) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss
