import torch
import torch.nn as nn

from blox import AttrDict, rmap
from blox.torch.dist import Gaussian, ProbabilisticModel, get_constant_parameter
from blox.torch.recurrent_modules import BaseCell, InitLSTMCell, CustomLSTMCell
from blox.torch.variational import setup_variational_inference
from blox.torch.losses import KLDivLoss2
from blox.torch.subnetworks import Predictor
""" The NoReinitVRNN classes remove the cumbersome separate initialization from the VRNN classes,
 unifying VRNN and ZeroVRNN. Unfortunately, NoReinitVRNN performance is untested and it is not recommended to use
 in practice until that happens. """


class NoReinitVRNNCell(BaseCell, ProbabilisticModel):
    # TODO try to implement this with a bunch of standard LSTMs, not CustomLSTM?
    # This would be likely a lot faster since we can use cuda lstm
    # However it's unclear how to implement sampling: z's have to be sampled sequentially, but there might be no
    # stochastic cuda lstm
    # It is possible that not sampling z's autoregressively will work, especially if multiple layers of latents are used
    # A fixed prior lstm would likely be very fast because it can be implemented with cuda lstm
    # It is possible to implement a stochastic lstm in pytorch JIT
    # It is also possible to implement the training with nn.LSTM and test time with manual unroll
    
    def __init__(self, hp, x_size, context_size):
        """ Implements the cell for the variational RNN (Chung et al., 2015)

        :param hp: an object with attributes:
            var_inf: can be ['standard', 'deterministic']
            prior_type: can be ['learned', 'fixed']
            nz_vae: # of dim in the vae latent
            builder: a LayerBuilderParams instance
            n_lstm_layers: number of layers in each LSTM network
            nz_mid_lstm: LSTM hidden size
        :param x_size: the size of the modelled data
        :param context_size: the size of additional context that is fed every step
        :param reinit_size: the size of information used to initialize the lstm
        """
        self._hp = hp
        BaseCell.__init__(self)
        ProbabilisticModel.__init__(self)
        inf_inp_dim = x_size
        prior_inp_dim = x_size
        
        self.inf_lstm = CustomLSTMCell(hp,
                                       input_size=x_size + context_size,
                                       output_size=inf_inp_dim)
        self.gen_lstm = CustomLSTMCell(hp,
                                       input_size=hp.nz_vae + context_size,  # pred_input_size,
                                       output_size=x_size)
        
        # TODO make a more expressive prior
        self.inf, self.prior = setup_variational_inference(hp, prior_inp_dim=prior_inp_dim, inf_inp_dim=inf_inp_dim)
    
    def forward(self, context=None, x_prime=None, more_context=None, z=None):
        """

        :param x: observation at current step
        :param context: to be fed at each timestep
        :param x_prime: observation at next step
        :param more_context: also to be fed at each timestep.
        :param z: (optional) if not None z is used directly and not sampled
        :return:
        """
        # TODO to get rid of more_context, make an interface that allows context structures
        output = AttrDict()
        
        output.p_z = self.prior(torch.zeros_like(x_prime))  # the input is only used to read the batchsize atm
        if x_prime is not None:
            output.q_z = self.inf(self.inf_lstm(x_prime, context, more_context).output)
        
        if z is None:
            if self._sample_prior:
                z = Gaussian(output.p_z).sample()
            else:
                z = Gaussian(output.q_z).sample()
        
        pred_input = [z, context, more_context]
        
        output.x = self.gen_lstm(*pred_input).output
        return output
    
    def reset(self):
        self.gen_lstm.reset()
        self.inf_lstm.reset()


class NoReinitVRNN(nn.Module):
    """ Implements the variational RNN (Chung et al., 2015)
    The variational RNN can be used to model sequences of high-dimensional data. It is a sequential application
    of deep Variational Bayes (Kingma'14, Rezende'14)

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
        super().__init__()
        self._hp = hp
        
        # TODO add global context
        # TODO add sequence context
        self.lstm = NoReinitVRNNCell(hp, x_dim, context_dim).make_lstm()
        
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
        lstm_inputs = AttrDict(x_prime=x)
        if context is not None:
            lstm_inputs.update(more_context=context)
        
        outputs = self.lstm(inputs=lstm_inputs, length=output_length + conditioning_length)
        # The way the conditioning works now is by zeroing out the loss on the KL divergence and returning less frames
        # That way the network can pass the info directly through z. I can also implement conditioning by feeding
        # the frames directly into predictor. that would require passing previous frames to the VRNNCell and
        # using a fake frame to condition the 0th frame on.
        outputs = rmap(lambda ten: ten[:, conditioning_length:], outputs)
        outputs.conditioning_length = conditioning_length
        return outputs
    
    def loss(self, inputs, outputs, log_error_arr=False):
        losses = AttrDict()
        
        losses.kl = KLDivLoss2(self._hp.kl_weight) \
            (outputs.q_z, outputs.p_z, log_error_arr=log_error_arr)
        
        return losses
