import math
from functools import partial
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from blox import AttrDict
from blox.tensor.ops import broadcast_final, batch_apply, map_recursive, batchwise_index, \
    batchwise_assign, remove_spatial, concat_inputs
from blox.torch.layers import BaseProcessingNet, ConvBlockEnc, \
    ConvBlockDec, init_weights_xavier, get_num_conv_layers, ConvBlockFirstDec, ConvBlock
from blox.torch.losses import CELoss
from blox.torch.modules import AttrDictPredictor, SkipInputSequential, GetIntermediatesSequential
from blox.torch.ops import apply_linear
from blox.torch.ops import like, make_one_hot, mask_out
from blox.torch.recurrent_modules import BaseProcessingLSTM, \
    BidirectionalLSTM, BareLSTMCell
from torch import Tensor
from torch.distributions.one_hot_categorical import OneHotCategorical


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


class Encoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp
        if hp.builder.use_convs:
            self.net = ConvEncoder(hp)
        else:
            self.net = Predictor(hp, hp.state_dim, hp.nz_enc, num_layers=hp.builder.get_num_layers())
    
    def forward(self, input):
        if self._hp.use_convs and self._hp.use_skips:
            return self.net(input)
        else:
            return self.net(input), None
    
    
class ConvEncoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp
        
        n = hp.builder.get_num_layers(hp.img_sz)
        self.net = GetIntermediatesSequential(hp.skips_stride) if hp.use_skips else nn.Sequential()
        
        self.net.add_module('input', ConvBlockEnc(in_dim=hp.input_nc, out_dim=hp.ngf, normalization=None,
                                                  builder=hp.builder))
        for i in range(n - 3):
            filters_in = hp.ngf * 2 ** i
            self.net.add_module('pyramid-{}'.format(i),
                                ConvBlockEnc(in_dim=filters_in, out_dim=filters_in*2, normalize=hp.builder.normalize,
                                             builder=hp.builder))
        
        # add output layer
        self.net.add_module('head', nn.Conv2d(hp.ngf * 2 ** (n - 3), hp.nz_enc, 4))

        self.net.apply(init_weights_xavier)

    def forward(self, input):
        return self.net(input)


class Decoder(nn.Module):
    """ A thin wrapper class that decides which decoder to build """
    def __init__(self, hp, regress_actions):
        super().__init__()
        self._hp = hp
        self.regress_actions = regress_actions
        
        if hp.builder.use_convs:
            assert not (self._hp.add_weighted_pixel_copy & self._hp.pixel_shift_decoder)
            
            if self._hp.pixel_shift_decoder:
                self.net = PixelShiftDecoder(hp)
            elif self._hp.add_weighted_pixel_copy:
                self.net = PixelCopyDecoder(hp)
            else:
                self.net = ConvDecoder(hp)
                
        else:
            assert not self._hp.use_skips
            assert not self._hp.add_weighted_pixel_copy
            assert not self._hp.pixel_shift_decoder
            state_predictor = Predictor(hp, hp.nz_enc, hp.state_dim, num_layers=hp.builder.get_num_layers())
            self.net = AttrDictPredictor({'images': state_predictor})
            
        if self.regress_actions:
            self.act_net = Predictor(hp, hp.nz_enc, hp.n_actions)

    def forward(self, input, **kwargs):
        if not (self._hp.pixel_shift_decoder or self._hp.add_weighted_pixel_copy):
            kwargs.pop('pixel_source')
            
        if not self._hp.use_skips:
            kwargs.pop('skips')
            
        output = self.net(input, **kwargs)
        
        actions = torch.flatten(self.act_net(input), 1) if self.regress_actions else None
        if actions is not None and self._hp.action_activation is not None:
            actions = self._hp.action_activation(actions)
        output.actions = actions
        return output

    def decode_seq(self, inputs, encodings):
        """ Decodes a sequence of images given the encodings
        
        :param inputs:
        :param encodings:
        :param seq_len:
        :return:
        """
        
        def extend_to_seq(tensor):
            return tensor[:, None].expand([tensor.shape[0], encodings.shape[1]] + list(tensor.shape[1:])).contiguous()

        # TODO skip from the goal as well
        seq_skips = map_recursive(extend_to_seq, inputs.skips)
        pixel_source = [extend_to_seq(inputs.I_0), extend_to_seq(inputs.I_g)]

        decoder_inputs = AttrDict(input=encodings, skips=seq_skips, pixel_source=pixel_source)
        return batch_apply(decoder_inputs, self, separate_arguments=True)


class ConvDecoder(nn.Module):
    def __init__(self, hp):
        super().__init__()

        self._hp = hp
        n = get_num_conv_layers(hp.img_sz)
        self.net = SkipInputSequential(hp.skips_stride) if hp.use_skips else nn.Sequential()
        out_dim = hp.ngf * 2 ** (n - 3)
        self.net.add_module('net',
                            ConvBlockFirstDec(in_dim=hp.nz_enc, out_dim=out_dim, normalize=hp.builder.normalize,
                                              builder=hp.builder))
        
        for i in reversed(range(n - 3)):
            filters_out = hp.ngf * 2 ** i
            filters_in = filters_out * 2
            if self._hp.use_skips and (i+1) % hp.skips_stride == 0:
                filters_in = filters_in * 2
                
            self.net.add_module('pyramid-{}'.format(i),
                                ConvBlockDec(in_dim=filters_in, out_dim=filters_out, normalize=hp.builder.normalize,
                                             builder=hp.builder))


        self.head_filters_out = filters_out = hp.ngf
        filters_in = filters_out
        if self._hp.use_skips and 0 % hp.skips_stride == 0:
            filters_in = filters_in * 2
            
        self.net.add_module('additional_conv_layer', ConvBlockDec(in_dim=filters_in, out_dim=filters_out,
                                                                      normalization=None, activation=nn.Tanh(), builder=hp.builder))

        self.gen_head = ConvBlockDec(in_dim=filters_out, out_dim=hp.input_nc, normalization=None,
                                                 activation=nn.Tanh(), builder=hp.builder, upsample=False)

        self.net.apply(init_weights_xavier)
        self.gen_head.apply(init_weights_xavier)

    def forward(self, *args, **kwargs):
        output = AttrDict()
        output.feat = self.net(*args, **kwargs)
        output.images = self.gen_head(output.feat)
        return output
    

class PixelCopyDecoder(ConvDecoder):
    def __init__(self, hp, n_masks=None):
        super().__init__(hp)
        self.n_pixel_sources = 4 if hp.skip_from_parents else 2
        n_masks = n_masks or self.n_pixel_sources + 1
        
        self.mask_head = ConvBlockDec(in_dim=self.head_filters_out, out_dim=n_masks,
                                      normalization=None, activation=nn.Softmax(dim=1), builder=hp.builder,
                                      upsample=False)
        self.apply(init_weights_xavier)
        
    def forward(self, *args, pixel_source, **kwargs):
        output = super().forward(*args, **kwargs)

        output.pixel_copy_mask, output.images = self.mask_and_merge(output.feat, pixel_source + [output.images])
        return output
    
    # @torch.jit.script_method
    def mask_and_merge(self, feat, pixel_source):
        # type: (Tensor, List[Tensor]) -> Tuple[Tensor, Tensor]
        
        mask = self.mask_head(feat)
        candidate_images = torch.stack(pixel_source, dim=1)
        images = (mask.unsqueeze(2) * candidate_images).sum(dim=1)
        return mask, images


class PixelShiftDecoder(PixelCopyDecoder):
    def __init__(self, hp):
        self.n_pixel_sources = 4 if hp.skip_from_parents else 2  # Two input images + two parents for non-svg
        super().__init__(hp, n_masks=1 + self.n_pixel_sources * 2)

        self.flow_heads = nn.ModuleList([])
        for i in range(self.n_pixel_sources):
            self.flow_heads.append(ConvBlockDec(in_dim=self.head_filters_out, out_dim=2, normalization=None,
                                                activation=None, builder=hp.builder, upsample=False))
        
        self.apply(init_weights_xavier)
    
    @staticmethod
    def apply_flow(image, flow):
        """ Modified from
        https://github.com/febert/visual_mpc/blob/dev/python_visual_mpc/pytorch/goalimage_warping/goalimage_warper.py#L81
        """

        theta = image.new_tensor([[1, 0, 0], [0, 1, 0]]).reshape(1, 2, 3).repeat_interleave(image.size()[0], dim=0)
        identity_grid = F.affine_grid(theta, image.size())
        sample_pos = identity_grid + flow.permute(0, 2, 3, 1)
        image = F.grid_sample(image, sample_pos)
        return image
    
    def forward(self, *args, pixel_source, **kwargs):
        output = ConvDecoder.forward(self, *args, **kwargs)

        output.flow_fields = list([head(output.feat) for head in self.flow_heads])
        output.warped_sources = list([self.apply_flow(source, flow) for source, flow in
                                      zip(pixel_source, output.flow_fields)])
        
        _, output.images = self.mask_and_merge(
            output.feat, pixel_source + output.warped_sources + [output.images])
        return output


class Attention(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp
        time_cond_length = self._hp.max_seq_len if self._hp.one_hot_attn_time_cond else 1
        input_size = hp.nz_enc * 2 + time_cond_length if hp.timestep_cond_attention else hp.nz_enc * 2
        self.query_net = Predictor(hp, input_size, hp.nz_attn_key)
        self.attention_layers = nn.ModuleList([MultiheadAttention(hp) for _ in range(hp.n_attention_layers)])
        self.predictor_layers = nn.ModuleList([Predictor(hp, hp.nz_enc, hp.nz_attn_key, num_layers=2)
                                               for _ in range(hp.n_attention_layers)])
        self.out = nn.Linear(hp.nz_enc, hp.nz_enc)

    def forward(self, enc_demo_seq, enc_demo_key_seq, e_l, e_r, start_ind, end_ind, inputs, timestep=None):
        """Performs multi-layered, multi-headed attention."""

        if self._hp.forced_attention:
            return batchwise_index(enc_demo_seq, timestep[:,0].long()), None

        # Get (initial) attention key
        if self._hp.one_hot_attn_time_cond and timestep is not None:
            one_hot_timestep = make_one_hot(timestep.long(), self._hp.max_seq_len).float()
        else:
            one_hot_timestep = timestep
        args = [one_hot_timestep] if self._hp.timestep_cond_attention else []
        
        query = self.query_net(e_l, e_r, *args)
        
        # Attend
        s_ind, e_ind = (torch.floor(start_ind), torch.ceil(end_ind)) if self._hp.mask_inf_attention \
                                                                     else (inputs.start_ind, inputs.end_ind)
        norm_shape_k = query.shape[1:]
        norm_shape_v = enc_demo_seq.shape[2:]
        raw_attn_output, att_weights = None, None
        for attention, predictor in zip(self.attention_layers, self.predictor_layers):
            raw_attn_output, att_weights = attention(query, enc_demo_key_seq, enc_demo_seq, s_ind, e_ind,
                                                     forced_attention_step=timestep if self._hp.forced_attention else None)
            x = F.layer_norm(raw_attn_output, norm_shape_v)
            query = F.layer_norm(predictor(x) + query, norm_shape_k)  # skip connections around attention and predictor

        return apply_linear(self.out, raw_attn_output, dim=1), att_weights     # output non-normalized output of final attention layer


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

    def forward(self, q, k, v, start_ind, end_ind, forced_attention_step=None):
        batch_size, time = list(k.shape)[:2]
        latent_shape = list(v.shape[2:])

        # perform linear operation and split into h heads
        q = apply_linear(self.q_linear, q, dim=1).view(batch_size, self.n_heads, self.nz_k_i, *latent_shape[1:])
        k = apply_linear(self.k_linear, k, dim=2).view(batch_size, time, self.n_heads, self.nz_k_i, *latent_shape[1:])
        v = apply_linear(self.v_linear, v, dim=2).view(batch_size, time, self.n_heads, self.nz_v_i, *latent_shape[1:])

        # compute masked, multi-headed attention
        vals, att_weights = self.attention(q, k, v, self.nz_k_i, start_ind, end_ind, self.dropout, forced_attention_step)

        # concatenate heads and put through final linear layer
        concat = vals.contiguous().view(batch_size, *latent_shape)
        return apply_linear(self.out, concat, dim=1), att_weights.mean(dim=-1)

    def attention(self, q, k, v, nz_k, start_ind, end_ind, dropout=None, forced_attention_step=None):

        def tensor_product(key, sequence):
            dims = list(range(len(list(sequence.shape)))[3:])
            return (key[:, None] * sequence).sum(dim=dims)

        attn_scores = tensor_product(q, k) / math.sqrt(nz_k) * self.temperature
        attn_scores = MultiheadAttention.mask_out(attn_scores, start_ind, end_ind)
        attn_scores = F.softmax(attn_scores, dim=1)

        if forced_attention_step is not None:
            scores_f = torch.zeros_like(attn_scores)
            batchwise_assign(scores_f, forced_attention_step[:, 0].long(), 1.0)

        scores = scores_f if forced_attention_step is not None else attn_scores
        if dropout is not None and dropout.p > 0.0:
            scores = dropout(scores)

        return (broadcast_final(scores, v) * v).sum(dim=1), attn_scores

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
        seq = seq.view(sh[:2] + [-1])
    
        if self.add_time:
            time = like(torch.arange, seq)(seq.shape[1])[None, :, None].repeat([sh[0], 1, 1])
            seq = torch.cat([seq, time], dim=2)

        proc_seq = self.run_net(seq)
        proc_seq = proc_seq.view(sh)
        return proc_seq


class ConvSeqEncodingModule(SeqEncodingModule):
    def build_network(self, input_size, hp):
        kernel_size = hp.conv_inf_enc_kernel_size
        assert kernel_size % 2 != 0     # need uneven kernel size for padding
        padding = int(np.floor(kernel_size / 2))
        n_layers = hp.conv_inf_enc_layers
        block = partial(ConvBlock, d=1, kernel_size=kernel_size, padding=padding)
        self.net = BaseProcessingNet(input_size, hp.nz_mid, hp.nz_enc, n_layers, hp.builder, block=block)
        
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


class AttnKeyEncodingModule(SeqEncodingModule):
    def build_network(self, input_size, hp):
        self.net = Predictor(hp, input_size, hp.nz_attn_key, num_layers=1)

    def forward(self, seq):
        return batch_apply(seq.contiguous(), self.net)


class RecurrentPolicyModule(SeqEncodingModule):
    def __init__(self, hp, input_size, output_size, add_time=True):
        super().__init__(hp, False)
        self.hp = hp
        self.output_size = output_size
        self.net = BaseProcessingLSTM(hp, input_size, output_size)

    def build_network(self, input_size, hp):
        pass

    def forward(self, seq):
        sh = list(seq.shape)
        seq = seq.view(sh[:2] + [-1])
        proc_seq = self.run_net(seq)
        proc_seq = proc_seq.view(sh[:2] + [self.output_size] + sh[3:])
        return proc_seq


class LengthPredictorModule(nn.Module):
    """Predicts the length of a segment given start and goal image encoding of that segment."""
    def __init__(self, hp):
        super().__init__()
        self._hp = hp
        self.p = Predictor(hp, hp.nz_enc * 2, hp.max_seq_len)

    def forward(self, e0, eg):
        """Returns the logits of a OneHotCategorical distribution."""
        output = AttrDict()
        output.seq_len_logits = remove_spatial(self.p(e0, eg))
        output.seq_len_pred = OneHotCategorical(logits=output.seq_len_logits)
        
        return output
    
    def loss(self, inputs, model_output):
        losses = AttrDict()
        losses.len_pred = CELoss(self._hp.length_pred_weight)(model_output.seq_len_logits, inputs.end_ind)
        return losses


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


class SumTreeHiddenStatePredictorModel(HiddenStatePredictorModel):
    """ A HiddenStatePredictor for tree morphologies. Averages parents' hidden states """

    def forward(self, hidden1, hidden2, *inputs):
        hidden_state = hidden1 + hidden2
        return super().forward(hidden_state, *inputs)


class LinTreeHiddenStatePredictorModel(HiddenStatePredictorModel):
    """ A HiddenStatePredictor for tree morphologies. Averages parents' hidden states """
    def build_network(self):
        super().build_network()
        self.projection = nn.Sequential(nn.Linear(self.get_state_size() * 2, self.get_state_size()))

    def forward(self, hidden1, hidden2, *inputs):
        hidden_state = self.projection(concat_inputs(hidden1, hidden2))
        return super().forward(hidden_state, *inputs)


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


class ActionConditioningWrapper(nn.Module):
    def __init__(self, hp, net):
        super().__init__()
        self.net = net
        self.ac_net = Predictor(hp, hp.nz_enc + hp.n_actions, hp.nz_enc)

    def forward(self, input, actions):
        net_outputs = self.net(input)
        padded_actions = torch.nn.functional.pad(actions, (0, 0, 0, net_outputs.shape[1] - actions.shape[1], 0, 0))
        net_outputs = batch_apply(torch.cat([net_outputs, broadcast_final(padded_actions, input)], dim=2), self.ac_net)
        return net_outputs

