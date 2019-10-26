import torch


def ten2ar(tensor):
    return tensor.detach().cpu().numpy()


def ar2ten(array, device):
    return torch.from_numpy(array).to(device)