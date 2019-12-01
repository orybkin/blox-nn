import torch

from blox import AttrDict
from blox.torch.dist import Gaussian, ProbabilisticModel
from blox.torch.recurrent_modules import BaseCell, InitLSTMCell
from blox.torch.variational import setup_variational_inference


class SVGCell(BaseCell, ProbabilisticModel):
    # TODO is is cleaner to implement this with a bunch of standard LSTMs, not CustomLSTM
    
    def __init__(self, hp, x_size, context_size, reinit_size):
        """ Implements the cell for SVG (Denton & Fergus, 2018)
        
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
    
    def init_state(self, first_x, context, more_context):
        """ Initi"""
        self.inf_lstm.init_state(context)
        self.gen_lstm.init_state(context)
        
        # Note: this might be unnecessary since the first frame is already provided above
        if more_context is not None:
            more_context = more_context[:, 0]
        self.inf_lstm(first_x, context[:, 0], more_context)
    
    def forward(self, x, context=None, x_prime=None, more_context=None):
        """
        
        :param x: observation at current step
        :param context: to be fed at each timestep
        :param x_prime: observation at next step
        :param more_context: also to be fed at each timestep.
        :return:
        """
        # TODO make an interface that allows context structures to get rid of more_context
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
        output.x_t = self.gen_lstm(*pred_input).output
        return output
    
    def reset(self):
        pass
