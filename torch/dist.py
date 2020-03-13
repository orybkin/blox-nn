from contextlib import contextmanager

import numpy as np
import torch
from torch import distributions

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


def get_constant_parameter(init_log_value, learn):
    return torch.nn.Parameter(torch.full((1,), init_log_value)[0], requires_grad=learn)


class Distribution():
    def nll(self, x):
        raise NotImplementedError
    
    def sample(self, x):
        raise NotImplementedError
    
    def kl_divergence(self, x):
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
    

class Beta(Distribution):
    """ A Beta distribution defined on [0,1]"""
    def __init__(self, mu, log_nu):
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


class Gaussian(Distribution):
    """ Represents a gaussian distribution """
    # TODO: implement a dict conversion function
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
        
    def sample(self):
        return self.mu + self.sigma * torch.randn_like(self.mu)

    def kl_divergence(self, other):
        """Here self=q and other=p and we compute KL(q, p)"""
        return (other.log_sigma - self.log_sigma) + (self.sigma ** 2 + (self.mu - other.mu) ** 2) \
                                                     / (2 * other.sigma ** 2) - 0.5

    def nll(self, x):
        # Negative log likelihood (probability)
        return 0.5 * torch.pow((x - self.mu) / self.sigma, 2) + self.log_sigma + 0.5 * np.log(2 * np.pi)
    
    def optimal_variance_nll(self, x):
        """ Computes the NLL of a gaussian with the optimal (constant) variance for these data """
        
        sigma = ((x - self.mu) ** 2).mean().sqrt()
        return Gaussian(mu=self.mu, sigma=sigma).nll(x)
    
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
    

class SequentialGaussian_SharedPQ(Distribution):
    """ stacks two Gaussians """
    def __init__(self, g1, z1, g2):
        """

        """
        self.g1 = g1
        self.g2 = g2
        self.z1 = z1
        assert z1.shape == g1.shape
        self.shared_dims = None     # how many shape dimensions are shared
        self._update_shared_dims()

    def sample(self):
        """
        sample z2 and concatentate with z1
        :return:
        """
        return torch.cat([self.z1, self.g2.sample()], dim=1)

    def kl_divergence(self, other):
        return self.g1.kl_divergence(other.g1)

    @property
    def shape(self):
        self._update_shared_dims()
        return self.g1.shape[:self.shared_dims]

    @property
    def mu(self):
        return self.g1.mu

    @staticmethod
    def stack(*argv, dim):
        return SequentialGaussian_SharedPQ._combine(torch.stack, *argv, dim=dim)

    @staticmethod
    def cat(*argv, dim):
        return SequentialGaussian_SharedPQ._combine(torch.cat, *argv, dim=dim)

    @staticmethod
    def _combine(fcn, *argv, dim):
        def fn_apply(inputs):
            mu, log_sigma = [], []
            for g in inputs:
                mu.append(g.mu)
                log_sigma.append(g.log_sigma)
            mu = fcn(mu, dim)
            log_sigma = fcn(log_sigma, dim)
            return Gaussian(mu, log_sigma)

        g1_list = [a.g1 for a in argv]
        g2_list = [a.g2 for a in argv]
        z1_list = [a.z1 for a in argv]
        return SequentialGaussian_SharedPQ(fn_apply(g1_list), fcn(z1_list, dim=dim), fn_apply(g2_list))

    def view(self, shape):
        # assume that this shape does not include the last dimensions
        self._update_shared_dims()
        self.g1 = self.g1.view(shape + list(self.g1.shape[self.shared_dims:]))
        self.g2 = self.g2.view(shape + list(self.g2.shape[self.shared_dims:]))
        self.z1 = self.z1.view(shape + list(self.z1.shape[self.shared_dims:]))
        return self

    def __getitem__(self, item):
        return SequentialGaussian_SharedPQ(self.g1[item], self.z1[item], self.g2[item])

    def _update_shared_dims(self):
        shared_dims = 0
        for i, j in zip(self.g1.shape, self.g2.shape):
            if i != j: break
            shared_dims += 1
        assert shared_dims is not 0  # need at least one shared dim between the Gaussians
        self.shared_dims = shared_dims


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