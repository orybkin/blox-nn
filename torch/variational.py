from blox import AttrDict
from blox.torch.dist import Gaussian, ProbabilisticModel
from blox.torch.losses import KLDivLoss2
from blox.torch.subnetworks import Predictor
from blox.torch.dist import get_constant_parameter
from blox.torch.ops import batchwise_index, make_one_hot, get_dim_inds

import torch.nn as nn
import torch


class GaussianPredictor(Predictor):
    def __init__(self, hp, input_dim, gaussian_dim=None, spatial=True):
        if gaussian_dim is None:
            gaussian_dim = hp.nz_vae
            
        super().__init__(hp, input_dim, gaussian_dim * 2, spatial=spatial)
    
    def forward(self, *inputs):
        return Gaussian(super().forward(*inputs), concat_dim=1)


class ApproximatePosterior(GaussianPredictor):
    def __init__(self, hp, inp_dim):
        super().__init__(hp, inp_dim)


class LearnedPrior(GaussianPredictor):
    def __init__(self, hp, cond_dim):
        super().__init__(hp, cond_dim)


class FixedPrior(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

    def forward(self, cond, *args):  # ignored because fixed prior
        return Gaussian.get_unit_gaussian([cond.shape[0], self.hp.nz_vae] + list(cond.shape[2:]), cond.device)


def get_prior(hp, inp_dim):
    if hp.prior_type == 'learned':
        return LearnedPrior(hp, inp_dim)
    elif hp.prior_type == 'fixed':
        return FixedPrior(hp)


def setup_variational_inference(hp, x_dim=None, cond_dim=0, prior_inp_dim=None, inf_inp_dim=None):
    """ Creates the inference and the prior networks
    
    :param hp: an object with attributes:
        var_inf: can be ['standard', 'deterministic']
        prior_type: can be ['learned', 'fixed']
        nz_vae: # of dim in the vae latent
    :param x_dim:
    :param cond_dim:
    :return:
    """
    prior_inp_dim = prior_inp_dim or cond_dim
    inf_inp_dim = inf_inp_dim or cond_dim + x_dim
    
    if hp.var_inf == 'standard':
        inf = ApproximatePosterior(hp, inf_inp_dim)
        prior = get_prior(hp, prior_inp_dim)

    elif hp.var_inf == 'deterministic':
        inf = FixedPrior(hp)
        prior = FixedPrior(hp)

    return inf, prior
    

class CVAE(nn.Module, ProbabilisticModel):
    """ A simple conditional VAE (Sohn et al., 2015) class.
    
    A conditional VAE learns to model a variable x with a latent variable z given some context cond.
    Specifically, it learns p(x|z,cond) as well as p(z|cond), and an approximate inference distribution q(z|x,cond).
    It is possible to use this class as a simple VAE by setting cond_dim=0.
    
    By default, the CVAE is going to run in the inference mode, i.e. produce reconstructions of input x.
    To switch the CVAE to the sampling mode, use `with vae.prior_mode():` context or the vae.switch_to_prior(),
    vae.switch_to_inference() switches.
    
    The CVAE class contains loss computation for the KL divergence, that can be used as:
    out = vae(inp)
    loss = vae.loss(inp, out)
    """
    
    def __init__(self, hp, x_dim, cond_dim=0, generator=None):
        """
        
        :param hp: an object with attributes:
            var_inf: can be ['standard', 'deterministic']
            prior_type: can be ['learned', 'fixed']
            nz_vae: # of dim in the vae latent
            kl_weight: the weight on the KL-divergence loss (usually set to 1)
            learn_sigma: whether the sigma of the decoder is learned
            log_sigma: if the sigma is learned, this sets the initialization for it
        :param x_dim: the dimension of the data that are going to be modelled
        :param cond_dim: the dimension of the context
        :param generator (optional): a module that produces the x given the z and the context
        """
        self._hp = hp
        nn.Module.__init__(self)
        ProbabilisticModel.__init__(self)
        if cond_dim == 0:
            assert hp.prior_type == 'fixed'
        
        if generator is None:
            generator = Predictor(hp, input_dim=hp.nz_vae + cond_dim, output_dim=x_dim)
        self.gen = generator
        self.inf, self.prior = setup_variational_inference(hp, x_dim, cond_dim)
        
        self.log_sigma = get_constant_parameter(hp.log_sigma, hp.learn_sigma)
        
    def forward(self, x, cond=None):
        output = AttrDict()
        if cond is None:
            cond = torch.zeros_like(x)[..., :0]  # If no conditioning, put a placeholder
        
        output.q_z = self.inf(x, cond)
        output.p_z = self.prior(cond)  # the input is only used to read the batchsize atm
    
        if self._sample_prior:
            output.z = output.p_z.sample()
        else:
            output.z = output.q_z.sample()
    
        output.mu = self.gen(output.z, cond)
        
        return output
    
    def loss(self, inputs, outputs):
        losses = AttrDict()
        losses.kl = KLDivLoss2(self._hp.kl_weight)(outputs.q_z, outputs.p_z)

        return losses
