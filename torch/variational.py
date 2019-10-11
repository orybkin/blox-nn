from blox import AttrDict
from blox.torch.dist import Gaussian, UnitGaussian, SequentialGaussian_SharedPQ, ProbabilisticModel
from blox.torch.losses import KLDivLoss
from blox.torch.subnetworks import Predictor
from blox.torch.ops import broadcast_final
import torch.nn as nn


class GaussianPredictor(Predictor):
    def __init__(self, hp, input_dim, gaussian_dim=None, spatial=False):
        if gaussian_dim is None:
            gaussian_dim = hp.nz_vae
            
        super().__init__(hp, input_dim, gaussian_dim * 2, spatial=spatial)
    
    def forward(self, *inputs):
        # TODO remove .tensor()
        return Gaussian(super().forward(*inputs)).tensor()


class ApproximatePosterior(GaussianPredictor):
    def __init__(self, hp, x_dim, cond_dim):
        super().__init__(hp, x_dim + cond_dim)


class LearnedPrior(GaussianPredictor):
    def __init__(self, hp, cond_dim):
        super().__init__(hp, cond_dim)


class FixedPrior(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

    def forward(self, cond, *args):  # ignored because fixed prior
        return UnitGaussian([cond.shape[0], self.hp.nz_vae], self.hp.device).tensor()


class VariationalInference2LayerSharedPQ(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.q1 = GaussianPredictor(hp, hp.nz_enc * 3, hp.nz_vae * 2)
        self.q2 = GaussianPredictor(hp, hp.nz_vae + 2 * hp.nz_enc, hp.nz_vae2 * 2)  # inputs are two parents and z1

    def forward(self, e_l, e_r, e_tilde):
        g1 = self.q1(e_l, e_r, e_tilde)
        z1 = Gaussian(g1).sample()
        g2 = self.q2(z1, e_l, e_r)
        return SequentialGaussian_SharedPQ(g1, z1, g2)


class TwolayerPriorSharedPQ(nn.Module):
    def __init__(self, hp, p1, q_p_shared):
        super().__init__()
        self.p1 = p1
        self.q_p_shared = q_p_shared

    def forward(self, e_l, e_r):
        g1 = self.p1(e_l, e_r)
        z1 = Gaussian(g1).sample()
        g2 = self.q_p_shared(z1, e_l, e_r)  # make sure its the same order of arguments as in usage above!!

        return SequentialGaussian_SharedPQ(g1, z1, g2)


class AttentiveInference(nn.Module):
    def __init__(self, hp, q, attention, run=True):
        super().__init__()
        self._hp = hp
        self.run = run
        self.q = q
        self.attention = attention
        self.deterministic = isinstance(self.q, FixedPrior)
    
    def forward(self, inputs, e_l, e_r, start_ind, end_ind, timestep=None):
        if not self.run:
            return self.get_dummy(e_l)
        
        output = AttrDict()
        if self.deterministic:
            output.q_z = self.q(e_l)
            return output
        
        e_tilde, output.gamma = self.attention(inputs.inf_enc_seq, inputs.inf_enc_key_seq, e_l, e_r,
                                               start_ind, end_ind, inputs, timestep)
        output.q_z = self.q(e_l, e_r, e_tilde)
        return output
    
    def loss(self, q_z, p_z):
        if q_z.numel() == 0:
            return {}
        
        return AttrDict(kl=KLDivLoss(self._hp.kl_weight, breakdown=1)(q_z, p_z, store_raw=True))
    
    def get_dummy(self, e_l):
        raise NotImplementedError('do we need to run inference in this case?')
        # TODO do we need to run inference in this case?
        return AttrDict(q_z=self.q.get_dummy())


def get_prior(hp, cond_dim):
    if hp.prior_type == 'learned':
        return LearnedPrior(hp, cond_dim)
    elif hp.prior_type == 'fixed':
        return FixedPrior(hp)


def setup_variational_inference(hp, x_dim, cond_dim):
    if hp.var_inf == '2layer':
        inf = VariationalInference2LayerSharedPQ(hp)
        prior = TwolayerPriorSharedPQ(hp, get_prior(hp, cond_dim), inf.p_q_shared)

    elif hp.var_inf == 'standard':
        inf = ApproximatePosterior(hp, x_dim, cond_dim)
        prior = get_prior(hp, cond_dim)

    elif hp.var_inf == 'deterministic':
        inf = FixedPrior(hp)
        prior = FixedPrior(hp)

    return inf, prior
    

class CVAE(nn.Module, ProbabilisticModel):
    """ A simple conditional VAE (Sohn et al., 2015) class. """
    
    def __init__(self, hp, x_dim, cond_dim, generator=None):
        """
        
        :param hp: an object with attributes:
            var_inf: can be ['standard', 'deterministic']
            prior_type: can be ['learned', 'fixed']
            nz_vae: # of dim in the vae latent
        :param x_dim:
        :param cond_dim:
        :param generator:
        """
        self._hp = hp
        nn.Module.__init__(self)
        ProbabilisticModel.__init__(self)
        
        if generator is None:
            generator = Predictor(hp, input_dim=hp.nz_vae + cond_dim, output_dim=x_dim)
        self.gen = generator
        self.inf, self.prior = setup_variational_inference(hp, x_dim, cond_dim)
        
        # self.inf = GaussianPredictor(hp, input_dim=x_dim + cond_dim, gaussian_dim=hp.nz_vae)  # inference
        # self.prior = GaussianPredictor(hp, input_dim=cond_dim, gaussian_dim=hp.nz_vae)  # prior
        
    def forward(self, x, cond):
        output = AttrDict()
        
        output.q_z = self.inf(x, cond)
        output.p_z = self.prior(cond)  # the input is only used to read the batchsize atm
    
        if self._sample_prior:
            output.z = Gaussian(output.p_z).sample()
        else:
            output.z = Gaussian(output.q_z).sample()
    
        output.mu = self.gen(output.z, cond)
        
        return output
    
    def loss(self, inputs, outputs):
        losses = AttrDict()
        losses.kl = KLDivLoss(self._hp.kl_weight, breakdown=1)(outputs.q_z, outputs.p_z)

        return losses
    