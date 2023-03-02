# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from torch import optim as optim
import ast


def build_partial_optimizer(model, opt_keys: list, freeze_keys: list, optimizer_name: str, optimizer_params: dict):
    """ Make an optimizer for specific params in a model

    Args:
        model (nn.Module)
        opt_keys (list): keys of params that need to be optimized
        freeze_keys (list): keys of params that don't need grad
        optimizer_name (str): name of optimizer in torch.optim
        optimizer_params (dict): params(lr, weight_decay) that pass to optimizer

    Returns:
        optimizer
    """
    # TODO: 分层学习率 scale
    params = []
    keys = []
    for key, value in model.named_parameters():
        for opt_key in opt_keys:
            if opt_key in key:
                value.requires_grad_(True)
                params += [{"params": [value]}]
                keys += [key]
                continue
        for freeze_key in freeze_keys:
            if freeze_key in key:
                value.requires_grad_(False)

    optimizer = getattr(optim, optimizer_name)(params, **optimizer_params)

    return optimizer


def build_optimizer(train_cfg, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    model_params = set_weight_decay(model, skip, skip_keywords)

    optimizer = getattr(optim, train_cfg.OPTIMIZER.NAME)(model_params, **ast.literal_eval(train_cfg.OPTIMIZER.PARAMS))

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
