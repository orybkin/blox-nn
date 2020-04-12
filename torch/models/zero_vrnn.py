from blox import AttrDict, rmap
from blox.torch.models.vrnn import VRNN
from blox.torch.ops import pad


class ZeroVRNN(VRNN):
    def forward(self, x, output_length, conditioning_length, context=None):
        """

        :param x: the modelled sequence, batch x time x  x_dim
        :param length: the desired length of the output sequence. Note, this includes all conditioning frames except 1
        :param conditioning_length: the length on which the prediction will be conditioned. Ground truth data are observed
        for this length
        :param context: a context sequence. Prediction is conditioned on all context up to and including this moment
        :return:
        """
        x = pad(x, pad_front=1, dim=1)
        lstm_inputs = AttrDict(x_prime=x[:, 1:])
        if context is not None:
            context = pad(context, pad_front=1, dim=1)
            lstm_inputs.update(more_context=context[:, 1:])
        
        initial_inputs = AttrDict(x=x[:, :conditioning_length + 1])
        
        self.lstm.cell.init_state(x[:, 0], more_context=context)
        outputs = self.lstm(inputs=lstm_inputs, initial_seq_inputs=initial_inputs,
                            length=output_length + conditioning_length)
        outputs = rmap(lambda ten: ten[:, conditioning_length:], outputs)
        return outputs