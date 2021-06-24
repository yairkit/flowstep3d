import torch

def debugger_mode():
    import sys
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    elif gettrace():
        return True


def get_num_workers(num_workers):
    if debugger_mode():
        return 0
    else:
        return num_workers


def detach_tensor(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x