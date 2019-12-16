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
        :param x_size: the size of the modelled data
        :param context_size: the size of additional context that is fed every step
        :param reinit_size: the size of information used to initialize the lstm
        """
        self._hp = hp
        BaseCell.__init__(self)
        ProbabilisticModel.__init__(self)
        # inf_lstm outputs input_size + output_size because that's what inf expects, however, this is arbitrary
        self.inf_lstm = InitLSTMCell(hp,
                                     input_size=x_size + context_size,
                                     output_size=x_size + context_size,
                                     reset_input_size=reinit_size)
        self.gen_lstm = InitLSTMCell(hp,
                                     input_size=x_size + hp.nz_vae + context_size,  #pred_input_size,
                                     output_size=x_size,
                                     reset_input_size=reinit_size)
        # inf expects input_size + context_size in, while prior expects context_size.
        # TODO make a more expressive prior
        self.inf, self.prior = setup_variational_inference(hp, x_size, context_size)
    
    def init_state(self, first_x, context, more_context=None):
        """ Initializes the state of the LSTM. Can be used to pass global context. Also performs the first step of
        inference LSTM"""
        if context is not None:
            self.inf_lstm.init_state(context)
            self.gen_lstm.init_state(context)
        
        # Note: this might be unnecessary since the first frame is already provided above
        # TODO is there a way to get rid of this?
        if more_context is not None:
            more_context = more_context[:, 0]
        self.inf_lstm(first_x, context, more_context)
    
    def forward(self, x, context=None, x_prime=None, more_context=None):
        """
        
        :param x: observation at current step
        :param context: to be fed at each timestep
        :param x_prime: observation at next step
        :param more_context: also to be fed at each timestep.
        :return:
        """
        # TODO to get rid of more_context, make an interface that allows context structures
        output = AttrDict()
        if x_prime is None:
            x_prime = torch.zeros_like(x)  # Used when sequence isn't available
        
        output.q_z = self.inf(self.inf_lstm(x_prime, context, more_context).output)
        output.p_z = self.prior(x, x)  # the input is only used to read the batchsize atm
        
        if self._sample_prior:
            z = Gaussian(output.p_z).sample()
        else:
            z = Gaussian(output.q_z).sample()
        
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
    """
    """
    
    def __init__(self, hp, x_dim):
        # TODO make test time version
        super().__init__()
        self._hp = hp
        
        # TODO add global context
        # TODO add sequence context
        self.lstm = VRNNCell(hp, x_dim, 0, 0).make_lstm()
    
    def forward(self, x, length):
        lstm_inputs = AttrDict(x_prime=x[:, 1:])
        initial_inputs = AttrDict(x=x[:, 0])
        
        self.lstm.cell.init_state(initial_inputs.x)
        outputs = self.lstm(inputs=lstm_inputs, initial_inputs=initial_inputs, length=length)
        return outputs
    
    def loss(self, inputs, model_output, log_error_arr=False):
        losses = AttrDict()
        
        losses.kl = KLDivLoss(self._hp.kl_weight) \
            (model_output.q_z, model_output.p_z, reduction=[-1, -2], log_error_arr=log_error_arr)
        
        return losses
    