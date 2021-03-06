import torch
from blox import AttrDict
from blox.torch.dist import Gaussian
from blox.tensor.ops import get_dim_inds


class LossDict(AttrDict):
    pass


class Loss():
    def __init__(self, weight=1.0, breakdown=None):
        """
        
        :param weight: the balance term on the loss
        :param breakdown: if specified, a breakdown of the loss by this dimension will be recorded
        """
        self.weight = weight
        self.breakdown = breakdown
    
    def __call__(self, *args, weights=1, reduction='mean', log_error_arr=False, **kwargs):
        """

        :param estimates:
        :param targets:
        :return:
        """
        error = self.compute(*args, **kwargs) * weights
        
        if reduction == 'mean':
            value = error.mean()
        elif isinstance(reduction, list) or isinstance(reduction, tuple):
            value = error.sum(reduction).mean()
        else:
            raise NotImplementedError
        loss = LossDict(value=value, weight=self.weight)
        
        if self.breakdown is not None:
            reduce_dim = get_dim_inds(error)[:self.breakdown] + get_dim_inds(error)[self.breakdown+1:]
            loss.breakdown = error.detach().mean(reduce_dim)
        if log_error_arr:
            loss.error_mat = error.detach()
        return loss
    
    def compute(self, estimates, targets):
        raise NotImplementedError
    

class L2Loss(Loss):
    def compute(self, estimates, targets, activation_function=None):
        # assert estimates.shape == targets.shape, "Input {} and targets {} for L2 loss need to have identical shape!"\
        #     .format(estimates.shape, targets.shape)
        if activation_function is not None:
            estimates = activation_function(estimates)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=estimates.device, dtype=estimates.dtype)
        l2_loss = torch.nn.MSELoss(reduction='none')(estimates, targets)
        return l2_loss
    

class L1Loss(Loss):
    def compute(self, estimates, targets):
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=estimates.device, dtype=estimates.dtype)
        l1_loss = torch.nn.L1Loss(reduction='none')(estimates, targets)
        return l1_loss


class KLDivLoss(Loss):
    def compute(self, estimates, targets):
        kl_divergence = estimates.kl_divergence(targets)
        return kl_divergence


class KLDivLoss2:
    """ This defines a distributional KL divergence loss that requires less boilerplate """
    def __init__(self, weight=1.0, breakdown=None, free_nats_per_dim=0):
        """

        :param weight: the balance term on the loss
        :param breakdown: if specified, a breakdown of the loss by this dimension will be recorded
        """
        self.weight = weight
        self.breakdown = breakdown
        self.free_nats_per_dim = free_nats_per_dim
    
    def __call__(self, estimates, targets, weights=1, log_error_arr=False):
        """

        :param estimates:
        :param targets:
        :return:
        """
        error = estimates.kl_divergence(targets) * weights
        # Sum across latents and average across the batch
        reduction = get_dim_inds(estimates)[1:]
        value = error.sum(reduction).mean()
        # Apply free nats
        free_nats = torch.full([], self.free_nats_per_dim * estimates.shape[1]).to(value.device)
        value = torch.max(value, free_nats)
        
        loss = LossDict(value=value, weight=self.weight)
        
        if self.breakdown is not None:
            reduce_dim = get_dim_inds(error)[:self.breakdown] + get_dim_inds(error)[self.breakdown + 1:]
            loss.breakdown = error.detach().mean(reduce_dim)
        if log_error_arr:
            loss.error_mat = error.detach()
        return loss


class CELogitsLoss(Loss):
    compute = staticmethod(torch.nn.functional.cross_entropy)
    
    
class BCELogitsLoss(Loss):
    compute = staticmethod(torch.nn.functional.binary_cross_entropy_with_logits)


class NLL(Loss):
    # Note that cross entropy is an instance of NLL, as is L2 loss.
    def compute(self, estimates, targets):
        nll = estimates.nll(targets)
        return nll
    

class PenaltyLoss(Loss):
    def compute(self, val):
        """Computes weighted mean of val as penalty loss."""
        return val