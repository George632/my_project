# Copyright (c) Facebook, Inc. and its affiliates.
from .model_wftr import build_wftr

MODEL_FUNCS = {
    "wftr": build_wftr,
}


def build_model(cfg):
    model, processor = MODEL_FUNCS[cfg.Model.model_name](cfg)
    return model, processor
