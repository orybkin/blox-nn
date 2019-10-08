import torch
from blox import AttrDict
from blox.torch.dist import Gaussian
from blox.tensor.ops import get_dim_inds


class Loss():
    def __init__(self, weight=1.0, breakdown=None):
        """
        
        :param weight: the balance term on the loss
        :param breakdown: if specified, a breakdown of the loss by this dimension will be recorded
        """
        self.weight = weight
        self.breakdown = breakdown
    
    def __call__(self, *args, weights=1, reduction='mean', store_raw=False, **kwargs):
        """

        :param estimates:
        :param targets:
        :return:
        """
        error = self.compute(*args, **kwargs) * weights
        if reduction != 'mean':
            raise NotImplementedError
        loss = AttrDict(value=error.mean(), weight=self.weight)
        if self.breakdown is not None:
            reduce_dim = get_dim_inds(error)[:self.breakdown] + get_dim_inds(error)[self.breakdown+1:]
            loss.breakdown = error.detach().mean(reduce_dim)
        if store_raw:
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


class KLDivLoss(Loss):
    def compute(self, estimates, targets):
        kl_divergence = Gaussian(estimates).kl_divergence(Gaussian(targets))
        return kl_divergence


class CELoss(Loss):
    compute = staticmethod(torch.nn.functional.cross_entropy)
    

class PenaltyLoss(Loss):
    def compute(self, val):
        """Computes weighted mean of val as penalty loss."""
        return val


if __name__ == "__main__":
    from recursive_planning.rec_planner_utils.logger import Logger
    logger = Logger(log_dir="./summaries")
    dummy_estimates = torch.rand([32, 10, 3, 64, 64])
    dummy_targets = torch.rand([32, 10, 3, 64, 64])
    dummy_weights = torch.rand([32, 10, 3, 64, 64])

    loss = l2_loss(dummy_estimates, dummy_targets, dummy_weights, logger, 10)

    dummy_estimates = torch.rand([32, 10, 64])
    dummy_targets = torch.rand([32, 10, 64])
    dummy_weights = torch.rand([32, 10, 32])

    loss = kl_div_loss(dummy_estimates, dummy_targets, dummy_weights, logger, 10)
    print("Done!")
