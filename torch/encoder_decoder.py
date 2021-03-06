import numpy as np
from blox import AttrDict, batch_apply, rmap
from blox.tensor.ops import broadcast_final, get_dim_inds
from blox.torch.ops import unpackbits, combine_dim, packbits, find_extra_dim
from blox.torch.dist import get_constant_parameter, Gaussian, Categorical, Bernoulli, DiscreteLogisticMixture
from blox.torch.layers import ConvBlockEnc, init_weights_xavier, get_num_conv_layers, ConvBlockFirstDec, ConvBlockDec
from blox.torch.losses import NLL
from blox.torch.modules import GetIntermediatesSequential, AttrDictPredictor, ConstantUpdater, SkipInputSequential
from blox.torch.subnetworks import Predictor
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        
class ImageCategorical(Categorical):
    """ This converts the input image from -1..1 to 0..255. This is useful to represent an image as a categorical
    distribution over all pixel values. It is factorized over colors and spatial locations.
    
    It should be initialized with a tensor batch x pixel_values x image_dims, where pixel_valuse=256, the number of
    different values a pixel is allowed to take."""
    def nll(self, x):
        return super().nll((x + 1) * 127.5)

    @property
    def mle(self):
        return self.log_p.argmax(1).float() / 127.5 - 1

    @property
    def mean(self):
        template = torch.arange(256)
        p = self.p
        value = broadcast_final(template.to(p.device).float()[None], p)
        return (p * value.float()).sum(1) / 127.5 - 1


class ImageBitwiseCategorical(Bernoulli):
    """ This is useful to represent a bitwise distribution - a distribution over each bit in a tensor
    
    It should be initialized with a tensor batch x bits x image_dims, where bits=8, the number of bits needed to
    describe each channel value.
    """

    def nll(self, x):
        dim = find_extra_dim(x, self.log_p)
        x_bitwise = unpackbits(((x + 1) * 127.5).round().byte(), dim)
        
        nll = super().nll(x_bitwise.float())
        return combine_dim(nll, dim, dim + 2)

    @property
    def mle(self):
        log_p = self.log_p
        bits_mle = log_p > 0
        mle = packbits(bits_mle.byte(), 1).float()
        return mle / 127.5 - 1

    @property
    def mean(self):
        template = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1])
        p = self.p
        value = broadcast_final(template.to(p.device).float()[None], p)
        return (p * value.float()).sum(1) / 127.5 - 1


class ImageDLM(DiscreteLogisticMixture):
    def nll(self, x):
        return super().nll((x + 1) / 2)
    
    @property
    def mean(self):
        return super().mean * 2 - 1


class HalfSigmoid(nn.Module):
    def forward(self, x):
        l = int(x.shape[1] / 2)
        x1, x2 = x[:, :l], x[:, l:]
        return torch.cat([torch.sigmoid(x1), x2], 1)


class ProbabilisticConvDecoder(nn.Module):
    """ This is a wrapper over ConvDecoders that makes the output a distribution """
    def __init__(self, hp, decoder_net):
        super().__init__()
        self._hp = hp
        self.net = decoder_net
        
        if hp.decoder_distribution == 'gaussian':
            self.log_sigma = get_constant_parameter(np.log(self._hp.initial_sigma), hp.learn_beta)
            self.sigma_updater = ConstantUpdater(self.log_sigma, 20, 'decoder_sigma')
        elif 'categorical' or 'discrete' in hp.decoder_distribution:
            # Children not supported
            assert type(decoder_net) == ConvDecoder

            if self._hp.decoder_distribution == 'categorical':
                self.n_values_per_pixel = 256
                self.distribution = ImageCategorical
                activation = None
            elif self._hp.decoder_distribution == 'bitwise_categorical':
                self.n_values_per_pixel = 8
                self.distribution = ImageBitwiseCategorical
                activation = None
            elif self._hp.decoder_distribution == 'discrete_logistic_mixture':
                self.n_values_per_pixel = 10
                self.distribution = ImageDLM
                activation = HalfSigmoid()
                
            self.net.gen_head = ConvBlockDec(in_dim=self.net.gen_head.params.in_dim,
                                             out_dim=self.n_values_per_pixel * hp.input_nc,
                                             normalization=None, activation=activation, builder=hp.builder, upsample=False)

    def forward(self, *args, **kwargs):
        outputs = self.net(*args, **kwargs)
        
        if self._hp.decoder_distribution == 'gaussian':
            # The sigma has to be blown up because of reshaping that happens later
            outputs.distr = Gaussian(outputs.images, self.log_sigma * torch.ones_like(outputs.images))
        elif 'categorical' in self._hp.decoder_distribution:
            images = outputs.images
            images = images.reshape(images.shape[0:1] + (self.n_values_per_pixel, -1) + images.shape[2:])
            
            outputs.distr = self.distribution(log_p=images)
            # outputs.images = outputs.distr.mle
            outputs.images = outputs.distr.mean
        elif 'discrete_logistic_mixture' == self._hp.decoder_distribution:
            images = outputs.images
            images = images.reshape(images.shape[0:1] + (self.n_values_per_pixel, -1) + images.shape[2:])
            n = int(images.shape[1] / 2)

            outputs.distr = self.distribution(images[:, :n], images[:, n:])
            outputs.images = outputs.distr.mean
        
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
    """ The decoder handles variation in the kinds of data that need to be decoded
    
    This module requires a hyperparameter dictionary with the following hyperparameters:
            'use_skips'
            'dense_rec_type'
            'decoder_distribution'
            'add_weighted_pixel_copy'
            'pixel_shift_decoder'
            'initial_sigma'
            'learn_beta'
    """
    
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
            if 'pixel_source' in kwargs:
                kwargs.pop('pixel_source')
        
        if not self._hp.use_skips:
            if 'skips' in kwargs:
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
        
        return batch_apply(self, input=encodings, skips=seq_skips, pixel_source=pixel_source)
    
    def loss(self, inputs, outputs, extra_action=True, first_image=True, log_error_arr=False):
        loss_gt = inputs.traj_seq
        loss_pad_mask = inputs.pad_mask
        if not first_image:
            loss_gt = loss_gt[:, 1:]
            loss_pad_mask = loss_pad_mask[:, 1:]
        
        weights = broadcast_final(loss_pad_mask, inputs.traj_seq)
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
