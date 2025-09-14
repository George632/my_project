# Copyright (c) Facebook, Inc. and its affiliates.
import torch


def build_optimizer(cfg, model):

    params_with_decay = []
    params_without_decay = []
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            continue
        if cfg.Optimizer.filter_biases_wd and (len(param.shape) == 1 or name.endswith("bias")):
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)

    if cfg.Optimizer.filter_biases_wd:
        param_groups = [
            {"params": params_without_decay, "weight_decay": 0.0},
            {"params": params_with_decay, "weight_decay": float(cfg.Optimizer.weight_decay)},
        ]
    else:
        param_groups = [
            {"params": params_with_decay, "weight_decay": float(cfg.Optimizer.weight_decay)},
        ]
    optimizer = torch.optim.AdamW(param_groups, lr=float(cfg.Optimizer.base_lr))
    return optimizer
