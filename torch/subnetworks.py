import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blox.tensor.ops import broadcast_final, batchwise_assign, remove_spatial
from blox.torch.layers import BaseProcessingNet, ConvBlock
from blox.torch.ops import apply_linear
from blox.torch.ops import like, mask_out
from blox.torch.recurrent_modules import BaseProcessingLSTM, BidirectionalLSTM, BareLSTMCell


class Predictor(BaseProcessingNet):
    def __init__(self, hp, input_dim, output_dim, num_layers=None, detached=False, spatial=True,
                 final_activation=None, mid_size=None):
        self.spatial = spatial
        mid_size = hp.nz_mid if mid_size is None else mid_size
        if num_layers is None:
            num_layers = hp.n_processing_layers

        super().__init__(input_dim, mid_size, output_dim, num_layers=num_layers, builder=hp.builder,
                         detached=detached, final_activation=final_activation)

    def forward(self, *inp):
        out = super().forward(*inp)
        return remove_spatial(out, yes=not self.spatial)


class MultiheadAttention(nn.Module):
    def __init__(self, hp, dropout=0.0):
        super().__init__()
        self._hp = hp
        self.nz = hp.nz_enc
        self.nz_attn_key = hp.nz_attn_key
        self.n_heads = hp.n_attention_heads
        assert self.nz % self.n_heads == 0  # number of attention heads needs to evenly divide latent
        assert self.nz_attn_key % self.n_heads == 0  # number of attention heads needs to evenly divide latent
        self.nz_v_i = self.nz // self.n_heads
        self.nz_k_i = self.nz_attn_key // self.n_heads
        self.temperature = nn.Parameter(self._hp.attention_temperature * torch.ones(1)) if self._hp.learn_attn_temp \
            else self._hp.attention_temperature

        # set up transforms for inputs / outputs
        self.q_linear = nn.Linear(self.nz_attn_key, self.nz_attn_key)
        self.k_linear = nn.Linear(self.nz_attn_key, self.nz_attn_key)
        self.v_linear = nn.Linear(self.nz, self.nz)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.nz, self.nz)

    def forward(self, q, k, v, start_ind, end_ind, forced_attention_step=None, attention_weights=None):
        batch_size, time = list(k.shape)[:2]
        latent_shape = list(v.shape[2:])

        # perform linear operation and split into h heads
        q = apply_linear(self.q_linear, q, dim=1).view(batch_size, self.n_heads, self.nz_k_i, *latent_shape[1:])
        k = apply_linear(self.k_linear, k, dim=2).view(batch_size, time, self.n_heads, self.nz_k_i, *latent_shape[1:])
        v = apply_linear(self.v_linear, v, dim=2).view(batch_size, time, self.n_heads, self.nz_v_i, *latent_shape[1:])

        # compute masked, multi-headed attention
        vals, att_weights = self.attention(q, k, v, self.nz_k_i, start_ind, end_ind, self.dropout, forced_attention_step,
                                           attention_weights)

        # concatenate heads and put through final linear layer
        concat = vals.contiguous().view(batch_size, *latent_shape)
        return apply_linear(self.out, concat, dim=1), att_weights.mean(dim=-1)

    def attention(self, q, k, v, nz_k, start_ind, end_ind, dropout=None, forced_attention_step=None,
                  attention_weights=None):

        def tensor_product(key, sequence):
            dims = list(range(len(list(sequence.shape)))[3:])
            return (key[:, None] * sequence).sum(dim=dims)

        scores = tensor_product(q, k) / math.sqrt(nz_k) * self.temperature
        scores = MultiheadAttention.mask_out(scores, start_ind, end_ind)
        scores = F.softmax(scores, dim=1)

        if forced_attention_step is not None:
            scores = torch.zeros_like(scores)
            batchwise_assign(scores, forced_attention_step[:, 0].long(), 1.0)

        if attention_weights is not None:
            scores = attention_weights[..., None].repeat_interleave(scores.shape[2], 2)

        if dropout is not None and dropout.p > 0.0:
            scores = dropout(scores)

        return (broadcast_final(scores, v) * v).sum(dim=1), scores

    @staticmethod
    def mask_out(scores, start_ind, end_ind):
        # Mask out the frames that are not in the range
        _, mask = mask_out(scores, start_ind, end_ind, -np.inf)
        scores[mask.all(dim=1)] = 1  # When the sequence is empty, fill ones to prevent crashing in Multinomial
        return scores

    def log_outputs_stateful(self, step, log_images, phase, logger):
        if phase == 'train':
            logger.log_scalar(self.temperature, 'attention_softmax_temp', step, phase)


class SeqEncodingModule(nn.Module):
    def __init__(self, hp, add_time=True):
        super().__init__()
        self.hp = hp
        self.add_time = add_time
        self.build_network(hp.nz_enc + add_time, hp)
        
    def build_network(self, input_size, hp):
        """ This has to define self.net """
        raise NotImplementedError()

    def run_net(self, seq):
        """ Run the network here """
        return self.net(seq)

    def forward(self, seq):
        sh = list(seq.shape)
        for s in seq.shape[3:]:
            assert s == 1
        seq = seq.view(sh[:2] + [-1])
    
        if self.add_time:
            time = like(torch.arange, seq)(seq.shape[1])[None, :, None].repeat([sh[0], 1, 1])
            seq = torch.cat([seq, time], dim=2)

        proc_seq = self.run_net(seq)
        proc_seq = proc_seq.view(sh[:2] + [-1] + sh[3:])
        return proc_seq


class ConvSeqEncodingModule(SeqEncodingModule):
    def build_network(self, input_size, hp, out_dim=None):
        kernel_size = hp.conv_inf_enc_kernel_size
        assert kernel_size % 2 != 0     # need uneven kernel size for padding
        padding = int(np.floor(kernel_size / 2))
        block = partial(ConvBlock, d=1, kernel_size=kernel_size, padding=padding)
        out_dim = out_dim or hp.nz_enc
        self.net = BaseProcessingNet(input_size, hp.nz_mid, out_dim, hp.conv_inf_enc_layers, hp.builder, block=block)
        
    def run_net(self, seq):
        # 1d convolutions expect length-last
        proc_seq = self.net(seq.transpose(1, 2)).transpose(1, 2)
        return proc_seq


class RecurrentSeqEncodingModule(SeqEncodingModule):
    def build_network(self, input_size, hp):
        self.net = BaseProcessingLSTM(hp, input_size, hp.nz_enc)


class BidirectionalSeqEncodingModule(SeqEncodingModule):
    def build_network(self, input_size, hp):
        self.net = BidirectionalLSTM(hp, input_size, hp.nz_enc)


class HiddenStatePredictorModel(BareLSTMCell):
    """ A predictor module that has a hidden state. The hidden state is exposed in the forward function """
    def __init__(self, hp, input_dim, output_dim):
        super().__init__(hp, input_dim, output_dim)
        self.build_network()

    def build_network(self):
        pass

    def forward(self, hidden_state, *inputs):
        output = super().forward(*inputs, hidden_state=hidden_state)
        return output.hidden_state, output.output


class GeneralizedPredictorModel(nn.Module):
    """Predicts the list of output values with optionally different activations."""
    def __init__(self, hp, input_dim, output_dims, activations, detached=False):
        super().__init__()
        assert output_dims  # need non-empty list of output dims defining the number of output values
        assert len(output_dims) == len(activations)     # need one activation for every output dim
        self._hp = hp
        self.activations = activations
        self.output_dims = output_dims
        self.num_outputs = len(output_dims)
        self._build_model(hp, input_dim, detached)

    def _build_model(self, hp, input_dim, detached):
        self.p = Predictor(hp, input_dim, sum(self.output_dims), detached=detached)

    def forward(self, *inputs):
        net_outputs = self.p(*inputs)
        outputs = []
        current_idx = 0
        for output_dim, activation in zip(self.output_dims, self.activations):
            output = net_outputs[:, current_idx:current_idx+output_dim]
            if activation is not None:
                output = activation(output)
            if output_dim == 1:
                output = output.view(-1)      # reduce spatial dimensions for scalars
            outputs.append(output)
        outputs = outputs[0] if len(outputs) == 1 else outputs
        return outputs
