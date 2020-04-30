# -*- coding: utf-8 -*-

import os
from configparser import ConfigParser

import torch
from .models_moco import XvectorNet


def parse_splice_str(splice_str):
    splice_indexes = []
    model_left_context = 0
    model_right_context = 0
    for splice in splice_str.strip().split():
        splice_indexes.append([int(t) for t in splice.split(",")])
        model_left_context += max(0, -splice_indexes[-1][0])
        model_right_context += max(0, splice_indexes[-1][-1])

    return splice_indexes, (model_left_context, model_right_context)


def build_model(conf, model_type="XVECTOR", write_back=False):
    if isinstance(conf, ConfigParser):
        model_cfg = conf
    else:
        assert os.path.exists(conf), f"{conf} file not exit!"

        model_cfg = ConfigParser()
        model_cfg.read(conf)

    xvector_cfg = model_cfg[model_type]

    input_size = xvector_cfg.getint("input_size")
    output_size = xvector_cfg.getint("output_size")
    hidden_size = xvector_cfg.getint("hidden_size")
    embedding_size = xvector_cfg.getint("embedding_size")
    normalize = xvector_cfg.get("normalize")
    dropout = xvector_cfg.getfloat("dropout")
    splice_indexes = xvector_cfg.get("splice_indexes")
    splice_indexes, extra_context = parse_splice_str(splice_indexes)

    # write model config to file
    if write_back and isinstance(conf, str):
        with open(conf, "w") as fptr:
            model_cfg.write(fptr)

    model = XvectorNet(input_size,
                       hidden_size,
                       output_size,
                       embedding_size,
                       splice_indexes,
                       normalize=normalize,
                       p=dropout)

    return model


def momentum_update(model_q, model_k, beta=0.999):
    """ model_k = beta * model_k + (1 - beta) model_q """
    with torch.no_grad():
        param_k = model_k.state_dict()
        param_q = model_q.named_parameters()
        for n, q in param_q:
            if n in param_k:
                param_k[n].data.mul_(beta).add_(1 - beta, q.data)


def get_shuffle_ids(batch_size, device=None):
    """generate shuffle ids for ShuffleBN"""
    forward_ids = torch.randperm(batch_size).long()
    value = torch.arange(batch_size).long()
    backward_ids = torch.zeros(batch_size).long()
    backward_ids.index_copy_(0, forward_ids, value)

    if device is not None:
        forward_ids = forward_ids.to(device)
        backward_ids = backward_ids.to(device)

    return forward_ids, backward_ids
