import torch
import torch.nn as nn

from blox import AttrDict, batch_apply, rmap
from blox.torch.dist import Gaussian, ProbabilisticModel, get_constant_parameter
from blox.torch.recurrent_modules import BaseCell, InitLSTMCell, CustomLSTMCell
from blox.torch.variational import setup_variational_inference
from blox.torch.losses import KLDivLoss
from blox.torch.subnetworks import Predictor
from blox.torch.layers import ConcatSequential
from blox.torch.sequence_predictors import SeqPredictorFactory
from blox.torch.modules import Batched
from blox.tensor.ops import concat_inputs


class SRNNGeneratorCell(BaseCell):
    
    # TODO try to implement this with a bunch of standard LSTMs, not CustomLSTM?
    # This would be likely a lot faster since we can use cuda lstm
    # However it's unclear how to implement sampling: z's have to be sampled sequentially, but there might be no
    # stochastic cuda lstm
    # It is possible that not sampling z's autoregressively will work, especially if multiple layers of latents are used
    # A fixed prior lstm would likely be very fast because it can be implemented with cuda lstm
    # It is possible to implement a stochastic lstm in pytorch JIT
    # It is also possible to implement the training with nn.LSTM and test time with manual unroll
    
    def __init__(self, hp, x_dim, context_dim, prior):
        """ This cell defines the predictive path of a latent variable model, implemented with LSTMs
        
        :param hp: an object with attributes:
            var_inf: can be ['standard', 'deterministic']
            prior_type: can be ['learned', 'fixed']
            nz_vae: # of dim in the vae latent
            builder: a LayerBuilderParams instance
            n_lstm_layers: number of layers in each LSTM network
            nz_mid_lstm: LSTM hidden size
        :param x_dim: the size of the modelled data
        :param context_dim: the size of additional context that is fed every step
        :param reinit_size: the size of information used to initialize the lstm
        """
        self._hp = hp
        BaseCell.__init__(self)
        
        self.gen_lstm = CustomLSTMCell(hp,
                                       input_size=hp.nz_vae + context_dim,  #pred_input_size,
                                       output_size=x_dim)
        self.prior = prior
    
    def forward(self, context=None, more_context=None, z=None, batch_size=None):
        """
        
        :param x: observation at current step
        :param context: to be fed at each timestep
        :param x_prime: observation at next step
        :param more_context: also to be fed at each timestep.
        :param z: (optional) if not None z is used directly and not sampled
        :return:
        """
        # TODO to get rid of more_context, make an interface that allows context structures
        outputs = AttrDict()
        
        outputs.p_z = self.prior(self.gen_lstm.output.weight.new_zeros(batch_size))  # the input is only used to read the batchsize atm
        if z is None:
            z = Gaussian(outputs.p_z).sample()
    
        pred_input = [z, context, more_context]
    
        outputs.x = self.gen_lstm(*pred_input).output
        return outputs
    
    def reset(self):
        self.gen_lstm.reset()


class SequentialVB(nn.Module, ProbabilisticModel):
    """ This is a class for general sequential variational Bayes methods. For a particular simple implementation,
    see blox.torch.models.vrnn
    Sequential variational Bayes is a technique to model sequences of high-dimensional data. It is a sequential
    application of deep variational Bayes (Kingma'14, Rezende'14)

    :param hp: an object with attributes:
        var_inf: can be ['standard', 'deterministic']
        prior_type: can be ['learned', 'fixed']
        nz_vae: # of dim in the vae latent
        builder: a LayerBuilderParams instance
        n_lstm_layers: number of layers in each LSTM network
        nz_mid_lstm: LSTM hidden size
    :param x_dim: the number of dimensions of a data point
    """
    
    def __init__(self, hp, x_dim, context_dim=0):
        # TODO make test time version

        nn.Module.__init__(self)
        ProbabilisticModel.__init__(self)
        self._hp = hp
        
        # TODO add global context
        # TODO add sequence context

        # TODO make a more expressive prior
        inf, prior = setup_variational_inference(hp, prior_inp_dim=x_dim, inf_inp_dim=x_dim)
        
        # TODO implement convolutional + recurrent inference
        self.inference = ConcatSequential(
            SeqPredictorFactory(hp.inference_arch, hp, x_dim + context_dim, x_dim, bidirectional=hp.bidirectional),
            Batched(inf),
            dim=2,
        )
        
        self.generator = SRNNGeneratorCell(hp, x_dim, context_dim, prior).make_lstm()
        
        self.log_sigma = get_constant_parameter(hp.log_sigma, hp.learn_sigma)
    
    def forward(self, x, output_length, conditioning_length, context=None):
        """
        
        :param x: the modelled sequence, batch x time x  x_dim
        :param length: the desired length of the output sequence. Note, this includes all conditioning frames except 1
        :param conditioning_length: the length on which the prediction will be conditioned. Ground truth data are observed
        for this length
        :param context: a context sequence. Prediction is conditioned on all context up to and including this moment
        :return:
        """
        lstm_inputs = AttrDict()
        outputs = AttrDict()
        if context is not None:
            lstm_inputs.more_context = context
    
        if not self._sample_prior:
            outputs.q_z = self.inference(x, context)
            lstm_inputs.z = Gaussian(outputs.q_z).sample()
    
        outputs.update(self.generator(inputs=lstm_inputs,
                                      length=output_length + conditioning_length,
                                      static_inputs=AttrDict(batch_size=x.shape[0])))
        # The way the conditioning works now is by zeroing out the loss on the KL divergence and returning less frames
        # That way the network can pass the info directly through z. I can also implement conditioning by feeding
        # the frames directly into predictor. that would require passing previous frames to the VRNNCell and
        # using a fake frame to condition the 0th frame on.
        outputs = rmap(lambda ten: ten[:, conditioning_length:], outputs)
        outputs.conditioning_length = conditioning_length
        return outputs
    
    def loss(self, inputs, outputs, log_error_arr=False):
        losses = AttrDict()
        
        losses.kl = KLDivLoss(self._hp.kl_weight) \
            (outputs.q_z, outputs.p_z, reduction=[-1, -2], log_error_arr=log_error_arr)
        
        return losses
