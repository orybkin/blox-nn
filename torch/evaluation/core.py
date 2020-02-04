import torch
import numpy as np

from blox.tensor import ndim
from blox.torch.ops import get_dim_inds
from blox.torch.evaluation.ssim_metric import create_window, _ssim


class metric:
    """ A function decorator that adds an argument 'per_datum'. """
    
    def __init__(self, func):
        self.func = func
    
    def __call__(self, estimates, targets, per_datum=False):
        """

        :param estimates:
        :param targets:
        :param per_datum: If this is True, return a tensor of shape: [batch_size], otherwise: [1]
        :return:
        """
        if targets.size == 0:
            return 0
        
        error = self.func(estimates, targets)
        if per_datum:
            return np.mean(error, axis=get_dim_inds(error)[1:])
        else:
            return np.mean(error)
        

@metric
def psnr(estimates, targets, data_dims=3):
    # NOTE: PSNR is not dimension-independent. The number of dimensions which are part of the metric has to be specified
    # I.e 2 for grayscale, 3 for color images.
    estimates = (estimates + 1) / 2
    targets = (targets + 1)/2

    max_pix_val = 1.0
    tolerance = 0.001
    assert (0 - tolerance) <= np.min(targets) and np.max(targets) <= max_pix_val * (1 + tolerance)
    assert (0 - tolerance) <= np.min(estimates) and np.max(estimates) <= max_pix_val * (1 + tolerance)

    mse = (np.square(estimates - targets))
    mse = np.mean(mse, axis=get_dim_inds(mse)[-data_dims:])

    psnr = 10 * np.log(max_pix_val / mse) / np.log(10)
    if np.any(np.isinf(psnr)):
        import pdb; pdb.set_trace()
    return psnr


@metric
def mse(estimates, targets):
    return (np.square(estimates - targets))


@metric
@ndim.numpied
def ssim(img1, img2, window_size=11):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel).detach().cpu().numpy()


class EvaluationCounter(object):
    def __init__(self, max_level):
        self.max_level = max_level
        self.clear()
    
    def __call__(self, pred, gt, mask=None):
        """ Adds the metrics to the counter
        
        :param pred: an array (bs x) time x image_dims..
        :param gt: an array (bs x) time x image_dims..
        :param mask:
        :return:
        """
        # reshape = lambda x: x.reshape([x.shape[0], -1, 3] + list(x.shape[-2:]))
        # pred, gt = reshape(pred), reshape(gt)
        pred = self.prepare_images(pred)
        gt = self.prepare_images(gt)
        
        for i in range(pred.shape[0]):
            self.psnr_sum += psnr(pred[i], gt[i])
            self.ssim_sum += ssim(pred[i], gt[i])
            # self.ssim_sum += ssim(torch.from_numpy(pred[i]).cuda(), torch.from_numpy(gt[i]).cuda())
            self.count += 1
    
    def prepare_images(self, images):
        if len(images.shape) == 4:
            return images[None]
        
        return images
      
    @property
    def PSNR(self):
        return self.psnr_sum / max(1, self.count)

    @property
    def SSIM(self):
        return self.ssim_sum / max(1, self.count)
    
    def clear(self):
        self.psnr_sum = 0
        self.ssim_sum = 0
        self.count = 0


