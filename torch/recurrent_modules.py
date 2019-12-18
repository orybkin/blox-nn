import torch
import torch.nn as nn
from blox.tensor.ops import concat_inputs
from funcsigs import signature
from blox import AttrDict, rmap_list
from blox.basic_types import map_dict, listdict2dictlist, subdict, filter_dict
from blox.torch.dist import stack
from blox.torch.layers import BaseProcessingNet, FCBlock


# Note: this post has an example custom implementation of LSTM from which we can derive a ConvLSTM/TreeLSTM
# https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/


class CustomLSTM(nn.Module):
    def __init__(self, cell):
        super(CustomLSTM, self).__init__()
        self.cell = cell
    
    def forward(self, inputs, length, initial_inputs=None, static_inputs=None, initial_seq_inputs={}):
        """
        
        :param inputs: These are sliced by time. Time is the second dimension
        :param length: Rollout length
        :param initial_inputs: These are not sliced and are overridden by cell output
        :param initial_seq_inputs: These can contain partial sequences. Cell output is used after these end.
        :param static_inputs: These are not sliced and can't be overridden by cell output
        :return:
        """
        # NOTE! Unrolling the cell directly will result in crash as the hidden state is not being reset
        # Use this function or CustomLSTMCell.unroll if needed
        initial_inputs, static_inputs = self.assert_begin(inputs, initial_inputs, static_inputs)

        step_inputs = initial_inputs.copy()
        step_inputs.update(static_inputs)
        lstm_outputs = []
        for t in range(length):
            step_inputs.update(map_dict(lambda x: x[:, t], inputs))  # Slicing
            step_inputs.update(map_dict(lambda x: x[:, t],
                                        filter_dict(lambda x: t < x[1].shape[1], initial_seq_inputs)))
            output = self.cell(**step_inputs)
            
            self.assert_post(output, inputs, initial_inputs, static_inputs)
            # TODO Test what signature does with *args
            autoregressive_output = subdict(output, output.keys() & signature(self.cell.forward).parameters)
            step_inputs.update(autoregressive_output)
            lstm_outputs.append(output)
        
        lstm_outputs = rmap_list(lambda *x: stack(x, dim=1), lstm_outputs)
            
        self.cell.reset()
        return lstm_outputs
    
    @staticmethod
    def assert_begin(inputs, initial_inputs, static_inputs):
        initial_inputs = initial_inputs or AttrDict()
        static_inputs = static_inputs or AttrDict()
        assert not (static_inputs.keys() & inputs.keys()), 'Static inputs and inputs overlap'
        assert not (static_inputs.keys() & initial_inputs.keys()), 'Static inputs and initial inputs overlap'
        assert not (inputs.keys() & initial_inputs.keys()), 'Inputs and initial inputs overlap'
        
        return initial_inputs, static_inputs
    
    @staticmethod
    def assert_post(output, inputs, initial_inputs, static_inputs):
        assert initial_inputs.keys() <= output.keys(), 'Initial inputs are not overridden'
        assert not ((static_inputs.keys() | inputs.keys()) & (output.keys())), 'Inputs are overridden'


class BaseProcessingLSTM(CustomLSTM):
    def __init__(self, hp, in_dim, out_dim):
        super().__init__(CustomLSTMCell(hp, in_dim, out_dim))
        
    def forward(self, input):
        """
        :param input: tensor of shape batch x time x channels
        :return:
        """
        return super().forward(AttrDict(cell_input=input), length=input.shape[1]).output


class BareProcessingLSTM(CustomLSTM):
    def __init__(self, hp, in_dim, out_dim):
        super().__init__(BareLSTMCell(hp, in_dim, out_dim))

    def forward(self, input, hidden_state, length=None):
        """
        :param input: tensor of shape batch x time x channels
        :return:
        """
        if length is None: length = input.shape[1]
        initial_state = AttrDict(hidden_state=hidden_state)
        outputs = super().forward(AttrDict(cell_input=input), length=length, initial_inputs=initial_state)
        return outputs


class BidirectionalLSTM(nn.Module):
    def __init__(self, hp, in_dim, out_dim):
        super().__init__()
        self.forward_lstm = CustomLSTM(CustomLSTMCell(hp, in_dim, out_dim))
        self.backward_lstm = CustomLSTM(CustomLSTMCell(hp, out_dim, out_dim))

    def forward(self, input):
        input_length = input.shape[1]

        def apply_and_reverse(lstm, input):
            return lstm.forward(AttrDict(cell_input=input), length=input_length).output.flip([1])

        return apply_and_reverse(self.backward_lstm, apply_and_reverse(self.forward_lstm, input))


class BaseCell(nn.Module):
    @staticmethod
    def unroll_lstm(lstm, step_fn, time):
        # NOTE! The CustomLSTM class should be used instead of this direct interface in most cases
        
        lstm_outputs = [step_fn(t) for t in range(time)]
        lstm.reset()
        return lstm_outputs
    
    def make_lstm(self):
        return CustomLSTM(self)


class CustomLSTMCell(BaseCell):
    def __init__(self, hp, input_size, output_size):
        """ An LSTMCell wrapper """
        super(CustomLSTMCell, self).__init__()
        
        hidden_size = hp.nz_mid_lstm
        n_layers = hp.n_lstm_layers
        
        # TODO make this a param dict
        self._hp = hp
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        # TODO use the LSTM class
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Linear(hidden_size, output_size)
        self.init_hidden()
        self.init_bias(self.lstm)
        
    @staticmethod
    def init_bias(lstm):
        for layer in lstm:
            for param in filter(lambda p: "bias" in p[0], layer.named_parameters()):
                name, bias = param
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    def init_hidden(self):
        # TODO Karl wrote some initializers that could be useful here
        self.initial_hidden = nn.Parameter(torch.zeros(self._hp.batch_size, self.get_state_size()))
        self.hidden = None

    def reset(self):
        # TODO make this trainable
        # calling set_hidden_var is necessary here since direct assignment is intercepted by nn.Module
        self.set_hidden_var(self.initial_hidden)
        
    def get_state_size(self):
        return self.hidden_size * self.n_layers * 2
    
    def var2state(self, var):
        """ Converts a tensor to a list of tuples that represents the state of the LSTM """
        
        var_layers = torch.chunk(var, self.n_layers, 1)
        return [torch.chunk(layer, 2, 1) for layer in var_layers]
    
    def state2var(self, state):
        """ Converts the state of the LSTM to one tensor"""
        
        layer_tensors = [torch.cat(layer, 1) for layer in state]
        return torch.cat(layer_tensors, 1)
    
    def forward(self, *cell_input, **cell_kwinput):
        """
        at every time-step the input to the dense-reconstruciton LSTM is a tuple of (last_state, e_0, e_g)
        :param cell_input:
        :param reset_indicator:
        :return:
        """
        # TODO allow ConvLSTM
        if cell_kwinput:
            cell_input = cell_input + list(zip(*cell_kwinput.items()))[1]

        if self.hidden is None:
            self.reset()
        
        cell_input = concat_inputs(*cell_input)
        inp_extra_dim = list(cell_input.shape[2:])  # This keeps trailing dimensions (should be all shape 1)
        embedded = self.embed(cell_input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        output = self.output(h_in)
        return AttrDict(output=output.view(list(output.shape) + inp_extra_dim))
    
    @property
    def hidden_var(self):
        if self.hidden is None:
            self.reset()
        return self.state2var(self.hidden)
    
    @hidden_var.setter
    def hidden_var(self, var):
        self.set_hidden_var(var)
    
    def set_hidden_var(self, var):
        self.hidden = self.var2state(var)


class InitLSTMCell(CustomLSTMCell):
    """ A wrapper around LSTM that conditionally initializes it with a neural network

     """
    
    def __init__(self, hp, input_size, output_size, reset_input_size):
        super(InitLSTMCell, self).__init__(hp, input_size, output_size)

        self.reset_input_size = reset_input_size
        
        if reset_input_size != 0:
            self.init_module = BaseProcessingNet(reset_input_size, self._hp.nz_mid,
                                                 self.get_state_size(), 1, hp.builder, FCBlock)
    
    def init_state(self, init_input):
        init_input = init_input.view(init_input.shape[0], self.reset_input_size)
        self.hidden_var = self.init_module(init_input)


class ReinitLSTMCell(InitLSTMCell):
    """ A wrapper around LSTM that allows to reinitialize the LSTM at specified timesteps.
    The reinitialization is done with an additional network that produces the hidden state given some input.
    The class allows to reinitialize different samples in the batch at different times.

     """
    
    def forward(self, cell_input, reset_input, reset_indicator):
        """
        :param inp: Input to the LSTM
        :param reset_input: Input to the reinitializer. Must be well-formed whenever reset_indicator is true
        :param reset_indicator: Tensor of shape [batch_size].
        Specifies which sample in the batch should be reinitialized
        """
        reset_input = reset_input.view(self._hp.batch_size, self.reset_input_size)
        # TODO only run when needed
        reinit_hidden = self.init_module(reset_input)
        if self.hidden is None:
            self.hidden_var = reinit_hidden
        else:
            self.hidden_var = torch.where(reset_indicator[:, None], reinit_hidden, self.hidden_var)
        return super().forward(cell_input)


class BareLSTMCell(CustomLSTMCell):
    """Exposes hidden state, takes initial hidden state input, returns final hidden state."""

    def forward(self, *cell_input, **cell_kwinput):
        assert 'hidden_state' in cell_kwinput   # BareLSTMCell needs hidden state input
        self.hidden_var = cell_kwinput.pop('hidden_state')
        output = super().forward(*cell_input, **cell_kwinput)
        output.hidden_state = self.hidden_var
        return output


class LSTMCellInitializer(nn.Module):
    """Base class for initializing LSTM states for start and end node."""
    def __init__(self, hp, cell):
        super().__init__()
        self._hp = hp
        self._cell = cell
        self._hidden_size = self._cell.get_state_size()

    def forward(self, *inputs):
        raise NotImplementedError


class ZeroLSTMCellInitializer(LSTMCellInitializer):
    """Initializes hidden to constant 0."""
    def forward(self, *inputs):
        def get_init_hidden():
            return inputs[0].new_zeros((inputs[0].shape[0], self._hidden_sz))
        return get_init_hidden(), get_init_hidden()


class MLPLSTMCellInitializer(LSTMCellInitializer):
    """Initializes hidden with MLP that gets start and goal image encodings as input."""
    def __init__(self, hp, cell, input_sz):
        super().__init__(hp, cell)
        from blox.torch.subnetworks import Predictor    # to avoid cyclic import
        self.net = Predictor(self._hp, input_sz, output_dim=2 * self._hidden_size, spatial=False,
                             num_layers=self._hp.init_mlp_layers, mid_size=self._hp.init_mlp_mid_sz)

    def forward(self, *inputs):
        hidden = self.net(*inputs)
        return hidden[:, :self._hidden_size], hidden[:, self._hidden_size:]


# class WarmupLSTMInitializer(ZeroLSTMCellInitializer):
#     """Runs the LSTM cell for given number of steps and returns internal state."""
#     def forward(self, *inputs):
#         hidden_start, hidden_end = super(*inputs)
#         z_dummy = hidden_start.new_zeros((hidden_start[0], self._hp.nz_vae))
#         for _ in range(self._hp.lstm_warmup_steps):
#             hidden_start, _ = \
#                 self.subgoal_pred(hidden_start, hidden_end, inputs[0], inputs[1], z_dummy)
#             hidden_end =




