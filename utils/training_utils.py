from typing import Optional, List

import torch
from click import Tuple

def get_optimizer(optim_name, params, lr, weight_decay=0.0, norm_weight_decay=None, model=None):
    if optim_name == 'sgd':
        optim = torch.optim.SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif optim_name == 'adam':
        optim = torch.optim.Adam(params, lr, eps=1e-07)
    elif optim_name == 'adamw':
        optim = torch.optim.AdamW(params, lr, eps=1e-07)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")
    return optim


def get_loss(loss_name, weights=None):
    print("INIT LOSS: with weights {}".format(weights))
    losses = {
        'cross_entropy': torch.nn.CrossEntropyLoss(weight=weights),
        'binary_cross_entropy': torch.nn.BCELoss(),
        'binary_cross_entropy_with_logit': torch.nn.BCEWithLogitsLoss(weight=weights),
        'mse': torch.nn.MSELoss(),
    }
    assert loss_name in losses.keys(
    ), f"Loss '{loss_name}' not supported, select one of the following: {list(losses.keys())}"
    return losses[loss_name]
