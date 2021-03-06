from contextlib import contextmanager

import numpy as np
import torch
import torch.utils.checkpoint
from torch import distributions
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from blox.torch.ops import find_extra_dim


def safe_entropy(dist, dim=None, eps=1e-12):
    """Computes entropy even if some entries are 0."""
    return -torch.sum(dist * safe_log_prob(dist, eps), dim=dim)


def safe_log_prob(tensor, eps=1e-12):
    """Safe log of probability values (must be between 0 and 1)"""
    return torch.log(torch.clamp(tensor, eps, 1 - eps))


def normalize(tensor, dim=1, eps=1e-7):
    norm = torch.clamp(tensor.sum(dim, keepdim=True), eps)
    return tensor / norm


def limit_exponent(tensor, min):
    """ Adds a constant to the tensor that brings the exponent of it to the desired min value.
    The constant is treated as a constant for autodiff (is detached)
    This is useful if one has a log variance and wants to impose a limit on the variance itself.  """
    
    diff = (tensor.exp() + min).log() - tensor
    result_tensor = tensor + diff.detach()
    
    return result_tensor


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)
    
    return result_tensor


def get_constant_parameter(init_log_value, learn=True):
    return torch.nn.Parameter(torch.full((1,), init_log_value)[0], requires_grad=learn)


class Distribution():
    def nll(self, x):
        raise NotImplementedError
    
    def sample(self, x):
        raise NotImplementedError
    
    def kl_divergence(self, other):
        raise NotImplementedError

    def cross_entropy(self, other):
        return self.entropy + self.kl_divergence(other)
    
    @property
    def entopy(self):
        raise NotImplementedError


class DiscreteGaussian(Distribution):
    """ Implements a discrete distribution specified by mu and sigma, such that the probability is proportional to
    the corresponding density function of a Gaussian. """
    
    # TODO this is not a great distribution because it suffers from the same instabilities as the Gaussian,
    #  and even more. Define a better distribution based on the hypergeometric or negative hypergeometric.
    
    def __init__(self, mu, log_sigma, range):
        self.mu = mu
        self.log_sigma = log_sigma
        
        vals = torch.arange(range).to(device=mu.device)
        gauss = torch.distributions.Normal(self.mu[..., None], self.log_sigma.exp()[..., None])
        cont_p = gauss.log_prob(vals).exp()
        disc_p = cont_p / cont_p.sum(-1, keepdim=True)
        self.categorical = torch.distributions.Categorical(probs=disc_p)
    
    def nll(self, x):
        return -self.categorical.log_prob(x)


class DiscreteLogistic(Distribution):
    """ IAFVAE """
    
    def __init__(self, mu, log_sigma, range=None):
        self.mu = mu
        self.log_sigma = log_sigma
    
    def cdf(self, x):
        return torch.sigmoid(x)
    
    def prob(self, x):
        # Return the probability mass
        
        # Use the memory efficient version
        # return torch.utils.checkpoint.checkpoint(self.prob_memory_efficient, x)
        
        # Below is an alternative computation provided for clarity
        mean = self.mu
        logscale = self.log_sigma
        binsize = 1 / 256.0
        
        mask_bottom = x == 0
        mask_top = x == 1
        
        scale = torch.exp(logscale)
        x = (torch.floor(x / binsize) * binsize - mean) / scale
        
        p = self.cdf(x + binsize / scale) - self.cdf(x)
        
        def masked_assign(dest, source, mask):
            """ Pytorch masking assign (dest[mask] = source[mask]) takes too much memory """
            mask = mask.float()
            return source * mask + dest * (1 - mask)

        # Edge cases
        p_bottom = self.cdf(x + binsize / scale)
        p = masked_assign(p, p_bottom, mask_bottom)
        # p[mask_bottom] = p_bottom[mask_bottom]
        p_top = 1 - self.cdf(x)
        p = masked_assign(p, p_top, mask_top)
        # p[mask_top] = p_top[mask_top]
        
        return p
    
    def prob_memory_efficient(self, x):
        # Return the probability mass
        # TODO: debug this, it gives good values but training dynamics is different fsr.
        mean = self.mu
        scale = self.log_sigma.exp()
        binsize = 1. / 256.0
    
        scaled_x = (torch.floor(x / binsize) * binsize - mean) / scale
        x_next = scaled_x + binsize / scale
        del scale
        p = self.cdf(x_next) - self.cdf(scaled_x)
    
        # Edge cases
        # TODO use torch.where (if it is memory efficient
        p_bottom = self.cdf(x_next)
        del x_next
        mask_bottom = (x == 0).float()
        p = p * (1 - mask_bottom)
        del mask_bottom
        p = p + p_bottom * ((x == 0).float())
        del p_bottom
    
        p_top = 1 - self.cdf(scaled_x)
        mask_top = (x == 1).float()
        del scaled_x
        p = p * (1 - mask_top)
        p = p + p_top * mask_top
        del p_top, mask_top
    
        return p

    def nll(self, x):
        p = self.prob(x)
        
        # Add epsilon for stability
        return -(p + 1e-7).log()
    
    @property
    def mean(self):
        return self.mu
    
    def to_dict(self):
        d = {'mu': self.mu, 'log_sigma': self.log_sigma}
        return d


class DiscreteLogisticMixture(DiscreteLogistic):
    def __init__(self, mu, log_sigma):
        """

        :param mu:
        :param log_sigma:
        :param n: number of elements in the mixture
        """
        self.mu = mu
        self.log_sigma = log_sigma
    
    def nll(self, x):
        mu, log_sigma = self.mu, self.log_sigma
        dim = find_extra_dim(x, mu)
        n = mu.shape[dim]
        
        x_tiled = x.unsqueeze(dim).repeat_interleave(n, dim)
        p = DiscreteLogistic(mu, log_sigma).prob(x_tiled).mean(dim)
        
        return -(p + 1e-7).log()
    
    @property
    def mean(self):
        return self.mu.mean(1)
    

class RescaledBeta(Distribution):
    """ A Beta distribution defined on [0,1]"""
    def __init__(self, mu, log_nu):
        # This distribution parametrizes beta in terms of location and scale
        # Unfortunately, this means we don't have control over what the values of a,b are,
        # while it is required that a,b>=1 for Beta to be defined on [0,1].
        # It is possible that I can figure out a loc-scale parametrization which enforces a,b>=1
        self.dist = distributions.Beta(mu * log_nu.exp(), (1 - mu) * log_nu.exp())

    def nll(self, x):
        x = self.rescale(x)
        # return (self.dist.concentration0 - 1) * x.log() + (self.dist.concentration1 - 1) * (1 - x).log() - self.log_norm
        return -self.dist.log_prob(x)
    
    def rescale(self, x, eps=1e-7):
        """ Rescale an input to be inside allowed range. Note: this operation is actually illegal but might be fine
        with small epsilon"""
        
        return x * (1 - eps * 2) + eps
    
    @property
    def log_norm(self):
        return self.dist._log_normalizer(self.dist.concentration0, self.dist.concentration1)


class LocScaleDistribution(Distribution):
    def __init__(self, mu, log_sigma=None, sigma=None, concat_dim=-1):
        """
        
        :param mu: the mean. this parameter should have the shape of the desired distribution
        :param log_sigma: If none, mu is divided into two chunks, mu and log_sigma
        """
        if log_sigma is None and sigma is None:
            if not isinstance(mu, torch.Tensor):
                import pdb; pdb.set_trace()
            mu, log_sigma = torch.chunk(mu, 2, concat_dim)
            
        self.mu = mu
        self._log_sigma = log_sigma
        self._sigma = sigma
        self.concat_dim = concat_dim

    @property
    def sigma(self):
        if self._sigma is None:
            self._sigma = self._log_sigma.exp()
        return self._sigma

    @property
    def log_sigma(self):
        if self._log_sigma is None:
            self._log_sigma = self._sigma.log()
        return self._log_sigma


class Gaussian(LocScaleDistribution):
    """ Represents a gaussian distribution """
    # TODO: implement a dict conversion function
    
    def sample(self):
        return self.mu + self.sigma * torch.randn_like(self.mu)

    def kl_divergence(self, other):
        """Here self=q and other=p and we compute KL(q, p)"""
        return (other.log_sigma - self.log_sigma) + (self.sigma ** 2 + (self.mu - other.mu) ** 2) \
                                                     / (2 * other.sigma ** 2) - 0.5

    def nll(self, x):
        # Negative log likelihood (probability)
        return 0.5 * torch.pow((x - self.mu) / self.sigma, 2) + self.log_sigma + 0.5 * np.log(2 * np.pi)
    
    @property
    def entropy(self):
        return (np.log(2 * np.pi * np.e) / 2) + self.log_sigma
    
    def optimal_variance_nll(self, x):
        """ Computes the NLL of a gaussian with the optimal (constant) variance for these data """
        
        sigma = ((x - self.mu) ** 2).mean().sqrt()
        return Gaussian(mu=self.mu, sigma=sigma).nll(x)

    def reparametrize(self, eps):
        """Reparametrizes noise sample eps with mean/variance of Gaussian."""
        return self.sigma * eps + self.mu

    @property
    def mean(self):
        return self.mu

    @property
    def shape(self):
        return self.mu.shape

    @staticmethod
    def stack(*argv, dim):
        return Gaussian._combine(torch.stack, *argv, dim=dim)

    @staticmethod
    def cat(*argv, dim):
        return Gaussian._combine(torch.cat, *argv, dim=dim)

    @staticmethod
    def _combine(fcn, *argv, dim):
        mu, log_sigma = [], []
        for g in argv:
            mu.append(g.mu)
            log_sigma.append(g.log_sigma)
        mu = fcn(mu, dim)
        log_sigma = fcn(log_sigma, dim)
        return Gaussian(mu, log_sigma)

    def view(self, shape):
        self.mu = self.mu.view(shape)
        self._log_sigma = self._log_sigma.view(shape)
        self._sigma = self.sigma.view(shape)
        return self

    def __getitem__(self, item):
        return Gaussian(self.mu[item], self._log_sigma[item])
 
    def tensor(self):
        return torch.cat([self.mu, self._log_sigma], dim=self.concat_dim)
    
    def to_dict(self):
        d = {'mu': self.mu}
        if self._log_sigma is None and self._sigma is None:
            raise ValueError
        if self._log_sigma is not None:
            d.update({'log_sigma': self._log_sigma})
        if self._sigma is not None:
            d.update({'sigma': self._sigma})
        return d
    
    @staticmethod
    def get_unit_gaussian(size, device):
        mu = torch.zeros(size, device=device)
        log_sigma = torch.zeros(size, device=device)
        return Gaussian(mu, log_sigma)


class OptimalVarianceGaussian(Gaussian):
    """ Technically not a distribution, however, it can compute NLL by adjusting it's variance to the datum at hand """
    
    def nll(self, x):
        return self.optimal_variance_nll(x)


class Laplacian(LocScaleDistribution):
    def nll(self, x):
        return torch.abs(x - self.mu) / self.scale + self.log_scale + np.log(2)
    
    @property
    def scale(self):
        # NOTE: this is NOT the std of the distribution, and self.sigma is NOT the std
        return self.sigma
    
    @property
    def log_scale(self):
        # NOTE: this is NOT the log std of the distribution, and self.log_sigma is NOT the log std
        return self.log_sigma
    
    def sample(self, x):
        raise NotImplementedError
    
    
class Categorical(Distribution):
    def __init__(self, p=None, log_p=None):
        # TODO log_p is actually unnormalized in most cases
        assert p is None or log_p is None
        
        self._log_p = log_p
        self._p = p
        
    @property
    def p(self):
        if self._p is not None:
            return self._p
        elif self._log_p is not None:
            # TODO use pytorch implementation?
            return self._log_p.exp() / self._log_p.exp().sum(1, keepdim=True)

    @property
    def log_p(self):
        if self._p is not None:
            return self._p.log()
        elif self._log_p is not None:
            return self._log_p
        
    @property
    def entropy(self):
        return safe_entropy(self.p, -1)
    
    def kl_divergence(self, other):
        return torch.sum(self.p * (self.p / other.p).log())
    
    def cross_entropy(self, other):
        return -torch.sum(self.p * other.log_p)
    
    def __add__(self, other):
        # Note, this is not a sum of random variables, but an equal mixture
        return Categorical((self.p + other.p) / 2)
    
    def nll(self, x):
        if self._log_p is not None:
            return SmartCrossEntropyLoss(reduction='none')(self._log_p, x.round().long())

    def to_dict(self):
        d = dict()
        if self._p is None and self._log_p is None:
            raise ValueError
        if self._p is not None:
            d.update({'p': self._p})
        if self._log_p is not None:
            d.update({'log_p': self._log_p})
        return d


class SmartCrossEntropyLoss(CrossEntropyLoss):
    """ This is a helper class that automatically finds which dimension is the classification dimension
    (as opposed to it always being dim=1) """
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # Find the dimension that has the distribution
        diff_idx = find_extra_dim(target, input)
        shape_target = target.shape
        target = target.reshape((-1,) + target.shape[diff_idx:])
        input = input.view((-1,) + input.shape[diff_idx:])
        
        loss = super().forward(input, target)
        
        if self.reduction == 'none':
            loss = loss.view(tuple(shape_target[:diff_idx]) + loss.shape[1:])
            
        return loss
    
    
class Bernoulli(Categorical):
    def nll(self, x):
        if self._log_p is not None:
            return F.binary_cross_entropy_with_logits(self._log_p, x, reduction='none')
        
    @property
    def p(self):
        if self._p is not None:
            return self._p
        elif self._log_p is not None:
            return self._log_p.sigmoid()


class ProbabilisticModel:
    def __init__(self):
        self._sample_prior = False
        
    def switch_to_prior(self):
        self._sample_prior = True

    def switch_to_inference(self):
        self._sample_prior = False
    

def get_fixed_prior(tensor, bs=None, dim=None):
    if dim is not None:
        return Gaussian(tensor.new_zeros(bs, dim, 1, 1), tensor.new_zeros(bs, dim, 1, 1))
    else:
        return Gaussian(torch.zeros_like(tensor.mu), torch.zeros_like(tensor.log_sigma))


def stack(inp, dim):
    if isinstance(inp[0], Gaussian):
        return Gaussian.stack(*inp, dim=dim)
    else:
        return torch.stack(inp, dim)
