import torch.nn as nn

from blox import batch_apply
from blox.torch.subnetworks import ConvSeqEncodingModule
from blox.tensor.ops import concat_inputs


class LSTMPredictor(nn.Module):
    def __init__(self, hp, in_dim, out_dim, **kwargs):
        super().__init__()
        self._hp = hp
        
        self.net = nn.LSTM(in_dim, hp.nz_mid_lstm, num_layers=hp.n_lstm_layers, batch_first=True, **kwargs)
        
        net_out_dim = hp.nz_mid_lstm
        if self.net.bidirectional:
            net_out_dim = net_out_dim * 2
        self.out_projection = nn.Linear(net_out_dim, out_dim)
    
    def forward(self, *inp):
        inp = concat_inputs(*inp, dim=2)
        
        n = self._hp.n_lstm_layers
        if self.net.bidirectional:
            n = n * 2
        c0 = inp.new_zeros(n, inp.shape[0], self._hp.nz_mid_lstm)
        h0 = inp.new_zeros(n, inp.shape[0], self._hp.nz_mid_lstm)
        
        out = self.net(inp, (c0, h0))[0]
        projected = batch_apply(self.out_projection, out.contiguous())
        return projected


class ConvSeqPredictor(ConvSeqEncodingModule):
    def __init__(self, hp, in_dim, out_dim, add_time=False):
        """
        
        :param hp:
        - conv_inf_enc_layers
        - conv_inf_enc_kernel_size
        :param in_dim:
        :param out_dim:
        :param add_time:
        """
        nn.Module.__init__(self)
        self.hp = hp
        self.add_time = add_time
        self.build_network(in_dim + add_time, hp, out_dim)


class ConvSeqLSTMPredictor(nn.Sequential):
    """ A predictor that combines temporal convolutions and LSTM """
    def __init__(self, hp, in_dim, out_dim, **kwargs):
        super().__init__()
        
        self.conv = ConvSeqPredictor(hp, in_dim, hp.nz_mid)
        self.lstm = LSTMPredictor(hp, hp.nz_mid, out_dim, **kwargs)


class SeqPredictorFactory:
    """ Constructs the appropriate seq predictor"""
    def __new__(cls, *args, **kwargs):
        type = args[0]
        if type == 'convseq':
            cl = ConvSeqPredictor
        if type == 'lstm':
            cl = LSTMPredictor
        if type == 'convseqlstm':
            cl = ConvSeqLSTMPredictor
        return cl(*args[1:], **kwargs)
