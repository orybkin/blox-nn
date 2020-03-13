import torch
import torch.nn as nn

from blox import AttrDict, batch_apply
from blox.torch.dist import Gaussian, ProbabilisticModel, get_constant_parameter
from blox.torch.recurrent_modules import BaseCell, InitLSTMCell
from blox.torch.variational import setup_variational_inference
from blox.torch.losses import KLDivLoss
from blox.torch.subnetworks import Predictor


class VRNNCell(BaseCell, ProbabilisticModel):
    # TODO try to implement this with a bunch of standard LSTMs, not CustomLSTM?
    # This would be likely a lot faster since we can use cuda lstm
    # However it's unclear how to implement sampling: z's have to be sampled sequentially, but there might be no
    # stochastic cuda lstm
    # It is possible that not sampling z's autoregressively will work, especially if multiple layers of latents are used
    # A fixed prior lstm would likely be very fast because it can be implemented with cuda lstm
    # It is possible to implement a stochastic lstm in pytorch JIT
    # It is also possible to implement the training with nn.LSTM and test time with manual unroll
    
    def __init__(self, hp, x_size, context_size, reinit_size):
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
        
        self.inf_lstm = InitLSTMCell(hp,
                                     input_size=x_size + context_size,
                                     output_size=inf_inp_dim,
                                     reset_input_size=reinit_size)
        self.gen_lstm = InitLSTMCell(hp,
                                     input_size=x_size + hp.nz_vae + context_size,  #pred_input_size,
                                     output_size=x_size,
                                     reset_input_size=reinit_size)
        
        # TODO make a more expressive prior
        self.inf, self.prior = setup_variational_inference(hp, prior_inp_dim=prior_inp_dim, inf_inp_dim=inf_inp_dim)
    
    def init_state(self, first_x, context=None, more_context=None):
        """ Initializes the state of the LSTM. Can be used to pass global context. Also performs the first step of
        inference LSTM """
        if context is not None:
            self.inf_lstm.init_state(context)
            self.gen_lstm.init_state(context)
        
        # TODO it would be extremely convenient to get rid of this by redefining the lstm to output one more frame
        # the loss would normally not be placed on that frame, but if it was, it would be a zero-frames conditioned VRNN
        # TODO the above needs to be done for an additional reason: currently the predictive model is not conditioned
        # on the context of the first frame, only the inference model is conditioned on it
        # (this is fine for action-conditioned prediction, where the first frame doesn't have context
        
        # Note: this might be unnecessary since the first frame is already provided above
        if more_context is not None:
            # TODO is there a way to get rid of this?
            more_context = more_context[:, 0]
        self.inf_lstm(first_x, context, more_context)
    
    def forward(self, x, context=None, x_prime=None, more_context=None, z=None):
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
        if x_prime is None:
            x_prime = torch.zeros_like(x)  # Used when sequence isn't available
        
        output.q_z = self.inf(self.inf_lstm(x_prime, context, more_context).output)
        output.p_z = self.prior(x)  # the input is only used to read the batchsize atm

        if z is not None:
            pass        # use z directly
        elif self._sample_prior:
            z = output.p_z.sample()
        else:
            z = output.q_z.sample()
        
        # Note: providing x might be unnecessary if it is provided in the init_state
        pred_input = [x, z, context, more_context]
        
        # x_t is fed back in as input (technically, it is unnecessary, however, the lstm is setup to observe a frame
        # every step because it observes one in the beginning).
        output.x = self.gen_lstm(*pred_input).output
        return output
    
    def reset(self):
        self.gen_lstm.reset()
        self.inf_lstm.reset()


class VRNN(nn.Module):
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
        self.lstm = VRNNCell(hp, x_dim, context_dim, 0).make_lstm()
        
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
        lstm_inputs = AttrDict(x_prime=x[:, 1:])
        if context is not None:
            lstm_inputs.update(more_context=context[:, 1:])
            
        initial_inputs = AttrDict(x=x[:, :conditioning_length])
        
        self.lstm.cell.init_state(x[:, 0], more_context=context)
        outputs = self.lstm(inputs=lstm_inputs, initial_seq_inputs=initial_inputs, length=output_length)
        return outputs
    
    def loss(self, inputs, model_output, log_error_arr=False):
        losses = AttrDict()
        
        losses.kl = KLDivLoss(self._hp.kl_weight) \
            (model_output.q_z, model_output.p_z, reduction=[-1, -2, -3, -4], log_error_arr=log_error_arr)
        
        return losses
