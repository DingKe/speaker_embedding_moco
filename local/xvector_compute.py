#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import subprocess

from argparse import ArgumentParser

import torch
import torch.nn as nn
import kaldi_io

from pytorch.nnet.core import np2tensor
from pytorch.nnet.checkpoint import load_checkpoint
from pytorch.nnet.utils import build_model


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="Launch Training")

    # Optional arguments for training
    parser.add_argument("--conf",
                        type=str,
                        default="conf/model.conf",
                        help="the path of model conf")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Sets the number of threads used for parallelizing CPU operations"
    )
    parser.add_argument("--use_gpu",
                        type=str,
                        default="false",
                        help="true/false")
    parser.add_argument("--device_id",
                        type=int,
                        default=0,
                        help="id of gpu to be used")
    parser.add_argument("--min_chunk_size", type=int, default=100, help="")
    parser.add_argument("--chunk_size", type=int, default=-1, help="")
    parser.add_argument("--pad_input", default='false', help="true/false")
    parser.add_argument("--model_type",
                        default='XVECTOR',
                        help="used for parsing config")
    parser.add_argument("ckpt_file")
    parser.add_argument("feat_rspecifier")
    parser.add_argument("xvector_wspecifier")
    return parser.parse_args()


args = parse_args()

true_strs = ['true', 't', 'yes', 'y']
if args.use_gpu.lower() in true_strs and torch.cuda.is_available():
    args.use_gpu = True
else:
    args.use_gpu = False

torch.set_num_threads(args.num_threads)
if args.use_gpu:
    device_id = args.device_id % torch.cuda.device_count()
    device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')

model = build_model(args.conf, model_type=args.model_type)
load_checkpoint(model, args.ckpt_file, map_location="cpu")
print('load model from %s' % args.ckpt_file, file=sys.stderr)
model.to(device)
model.prune()

left, right = model.extra_context

pad_input = True if args.pad_input.lower() in true_strs else False
min_chunk_size = max(args.min_chunk_size, left + right + 2)
chunk_size = args.chunk_size


def pad(feat, min_chunk_size):
    if len(feat) >= min_chunk_size:
        # no op
        return feat

    delta = min_chunk_size - len(feat)
    left = delta // 2
    right = delta - left

    return np.pad(feat, ((left, right), (0, 0)), mode="edge")


def extract(model, feat):
    feat = np.expand_dims(feat, 0)
    input = np2tensor(feat).to(device)
    embed_a, embed_b = model.extract_xvector_all(input)
    xvector = embed_b.cpu().numpy().squeeze(0)
    return xvector


feat_rspec = args.feat_rspecifier
xvector_wspec = args.xvector_wspecifier

read_cmd = f"copy-feats '{feat_rspec}' ark:- |"
feat_gen = kaldi_io.read_mat_ark(read_cmd)

write_cmd = f"| copy-vector ark:- {xvector_wspec}"
fptr = kaldi_io.open_or_fd(write_cmd)

with torch.no_grad():
    model.eval()
    for key, feat in feat_gen:
        if pad_input:
            feat = pad(feat, min_chunk_size)

        if len(feat) < min_chunk_size:
            print(f"{key} is shorter than {min_chunk_size}, skip it.",
                  file=sys.stderr)
            continue

        print(f"Process {key} with {len(feat)} frames", file=sys.stderr)

        if chunk_size <= 0:
            xvector = extract(model, feat)
        else:
            xvector_avg = 0
            num_frames = 0
            offset = 0
            while offset <  len(feat):
                end = offset + chunk_size
                if len(feat) - end < chunk_size: # flush the remainder
                    end = len(feat)
                sub_feat = feat[offset:end]
                if len(sub_feat) < min_chunk_size: continue
                xvector_avg = xvector_avg + extract(model,
                                                    sub_feat) * len(sub_feat)
                num_frames += len(sub_feat)
                offset = end
            xvector = xvector_avg / num_frames

        kaldi_io.write_vec_flt(fptr, xvector, key)
fptr.close()
