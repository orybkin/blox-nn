import numpy as np
from blox import AttrDict, batch_apply2, rmap
from blox.tensor.ops import broadcast_final, get_dim_inds
from blox.torch.dist import get_constant_parameter, Gaussian, Categorical
from blox.torch.layers import ConvBlockEnc, init_weights_xavier, get_num_conv_layers, ConvBlockFirstDec, ConvBlockDec
from blox.torch.losses import NLL
from blox.torch.modules import GetIntermediatesSequential, AttrDictPredictor, ConstantUpdater, SkipInputSequential
from blox.torch.subnetworks import Predictor
import torch
import torch.nn as nn
import torch.functional as F


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
        
        n = hp.builder.get_num_layers(hp.img_sz, hp.n_conv_layers)
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


class ConvDecoder(nn.Module):
    def __init__(self, hp):
        super().__init__()

        self._hp = hp
        n = get_num_conv_layers(hp.img_sz, hp.n_conv_layers)
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
        

class ProbabilisticConvDecoder(nn.Module):
    """ This is a wrapper over ConvDecoders that makes the output a distribution """
    def __init__(self, hp, decoder_net):
        super().__init__()
        self._hp = hp
        self.net = decoder_net
        
        if hp.decoder_distribution == 'gaussian':
            self.log_sigma = get_constant_parameter(np.log(self._hp.initial_sigma), hp.learn_beta)
            self.sigma_updater = ConstantUpdater(self.log_sigma, 20, 'decoder_sigma')
        elif hp.decoder_distribution == 'categorical':
            # Children not supported
            assert type(decoder_net) == ConvDecoder
            self.net.gen_head = ConvBlockDec(in_dim=self.net.gen_head.params.in_dim, out_dim=256 * hp.input_nc,
                                             normalization=None, activation=None, builder=hp.builder, upsample=False)

    def forward(self, *args, **kwargs):
        outputs = self.net(*args, **kwargs)
        
        if self._hp.decoder_distribution == 'gaussian':
            # The sigma has to be blown up because of reshaping that happens later
            outputs.distr = Gaussian(outputs.images, self.log_sigma * torch.ones_like(outputs.images))
        elif self._hp.decoder_distribution == 'categorical':
            images = outputs.images
            images = images.reshape(images.shape[0:1] + (256, 3) + images.shape[2:])
            
            outputs.distr = Categorical(log_p=images)
            outputs.images = images.argmax(1).float() / 127.5 - 1
        
        return outputs

    def nll(self, estimates, targets, weights=1, log_error_arr=False):
        """
        
        :param estimates: a distribution object
        """
        losses = AttrDict()
        
        criterion = NLL(self._hp.dense_img_rec_weight, breakdown=1)
        avg_inds = get_dim_inds(targets)[1:]
        losses.dense_img_rec = criterion(
            estimates, targets, weights=weights, reduction=avg_inds, log_error_arr=log_error_arr)
    
        return losses


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


class DecoderModule(ProbabilisticConvDecoder):
    """ The decoder handles variation in the kinds of data that need to be decoded """
    
    def __init__(self, hp, regress_actions):
        self._hp = hp
        decoder_net = self.build_decoder_net()
        super().__init__(hp, decoder_net)
        
        self.regress_actions = regress_actions
        if regress_actions:
            self.act_net = Predictor(hp, hp.nz_enc, hp.n_actions)
            self.act_log_sigma = get_constant_parameter(0, hp.learn_beta)
            self.act_sigma_updater = ConstantUpdater(self.act_log_sigma, 20, 'decoder_action_sigma')

    def build_decoder_net(self):
        hp = self._hp
        if self._hp.builder.use_convs:
            assert not (self._hp.add_weighted_pixel_copy & self._hp.pixel_shift_decoder)
            if self._hp.pixel_shift_decoder:
                decoder_net = PixelShiftDecoder(self._hp)
            elif self._hp.add_weighted_pixel_copy:
                decoder_net = PixelCopyDecoder(self._hp)
            else:
                decoder_net = ConvDecoder(self._hp)
        else:
            assert not self._hp.use_skips
            assert not self._hp.add_weighted_pixel_copy
            assert not self._hp.pixel_shift_decoder
            state_predictor = Predictor(hp, hp.nz_enc, hp.state_dim, num_layers=hp.builder.get_num_layers())
            decoder_net = AttrDictPredictor({'images': state_predictor})

        return decoder_net

    def forward(self, input, **kwargs):
        if not (self._hp.pixel_shift_decoder or self._hp.add_weighted_pixel_copy):
            kwargs.pop('pixel_source')
        
        if not self._hp.use_skips:
            kwargs.pop('skips')
        
        output = super().forward(input, **kwargs)
        
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
        
        # TODO skip from the goal as well
        extend_to_seq = lambda x: x[:, None][:, [0] * encodings.shape[1]].contiguous()
        seq_skips = rmap(extend_to_seq, inputs.skips)
        pixel_source = rmap(extend_to_seq, [inputs.I_0, inputs.I_g])
        
        return batch_apply2(self, input=encodings, skips=seq_skips, pixel_source=pixel_source)
    
    def loss(self, inputs, outputs, extra_action=True, first_image=True, log_error_arr=False):
        loss_gt = inputs.demo_seq
        loss_pad_mask = inputs.pad_mask
        if not first_image:
            loss_gt = loss_gt[:, 1:]
            loss_pad_mask = loss_pad_mask[:, 1:]
        
        weights = broadcast_final(loss_pad_mask, inputs.demo_seq)
        # Skip first frame
        losses = self.nll(outputs.distr, loss_gt[:, 1:], weights[:, 1:], log_error_arr)
        
        if self._hp.regress_actions:
            actions_pad_mask = inputs.pad_mask[:, :-1]
            loss_actions = outputs.actions
            if extra_action:
                loss_actions = loss_actions[:, :-1]
            
            weights = broadcast_final(actions_pad_mask, inputs.actions)
            losses.dense_action_rec = NLL(self._hp.dense_action_rec_weight) \
                (loss_actions, inputs.actions, weights=weights, reduction=[-1, -2], log_error_arr=log_error_arr)
        
        return losses
