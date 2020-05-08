# -*- coding: utf-8 -*-

import os
from configparser import ConfigParser

from .models import XvectorNet


def parse_splice_str(splice_str):
    splice_indexes = []
    model_left_context = 0
    model_right_context = 0
    for splice in splice_str.strip().split():
        splice_indexes.append([int(t) for t in splice.split(",")])
        model_left_context += max(0, -splice_indexes[-1][0])
        model_right_context += max(0, splice_indexes[-1][-1])

    return splice_indexes, (model_left_context, model_right_context)


def build_model(conf, model_type="XVECTOR", num_classes=0, write_back=False):
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
    metric = xvector_cfg.get("metric", "softmax")
    splice_indexes = xvector_cfg.get("splice_indexes")
    splice_indexes, extra_context = parse_splice_str(splice_indexes)

    num_classes_cfg = xvector_cfg.getint("num_classes")
    if num_classes_cfg <= 0:
        assert num_classes > 0, "Not valid num_classes provided!"
        num_classes_cfg = num_classes
        xvector_cfg["num_classes"] = str(num_classes_cfg)
    elif num_classes <= 0:
        num_classes = num_classes_cfg
    else:
        assert num_classes == num_classes_cfg, "Inconsistent num_classes between --num_classes and conf"

    # write model config to file
    if write_back and isinstance(conf, str):
        with open(conf, "w") as fptr:
            model_cfg.write(fptr)

    model = XvectorNet(input_size,
                       hidden_size,
                       output_size,
                       embedding_size,
                       num_classes,
                       splice_indexes,
                       normalize=normalize,
                       metric=metric,
                       p=dropout)

    return model
