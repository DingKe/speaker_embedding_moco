#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess
import numpy as np

from argparse import ArgumentParser
from configparser import ConfigParser

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter

from pytorch.datasets.dataqueue import DataQueue, Prefetcher
from pytorch.datasets.kaldi_xvector import KaldiXvector
from specaugment import SpecAugment
from pytorch.nnet.checkpoint import save_checkpoint, load_checkpoint
from pytorch.nnet.utils import parse_splice_str, build_model


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="Launch Training ")

    # Optional args for training
    parser.add_argument("--num_classes", type=int, default=0, help="# speaker")
    parser.add_argument("--seed",
                        type=int,
                        default=931111,
                        help="seed for random engine")
    parser.add_argument("--frames_per_epoch",
                        type=int,
                        default=1000000,
                        help="frames to train for each epoch")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=15,
                        help="epochs to train")
    parser.add_argument("--start_epoch",
                        type=int,
                        default=0,
                        help="start from this epoch")
    parser.add_argument("--exit_epoch",
                        type=int,
                        default=0,
                        help="exit at this epoch")
    parser.add_argument("--checkpoint_period",
                        type=int,
                        default=0,
                        help="save checkpoint every that samples")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1024,
                        help="batch size")
    parser.add_argument("--warmup_lr",
                        type=float,
                        default=0,
                        help="warmup learning rate")
    parser.add_argument("--initial_lr",
                        type=float,
                        default=1e-2,
                        help="initial learning rate")
    parser.add_argument("--final_lr",
                        type=float,
                        default=1e-3,
                        help="final learning rate")
    parser.add_argument("--linear_decay",
                        dest="linear_decay",
                        action="store_true",
                        help="if decay learning rate linearly")
    parser.add_argument("--min_chunk_size",
                        type=int,
                        default=200,
                        help="minimum chunk size for training")
    parser.add_argument("--max_chunk_size",
                        type=int,
                        default=500,
                        help="maximum chunk size for training")
    parser.add_argument("--conf",
                        type=str,
                        default="conf/model.conf",
                        help="the path of model conf")
    parser.add_argument("--model_type",
                        type=str,
                        default="XVECTOR",
                        help="the type of model")

    # Positional args for training
    parser.add_argument("model_dir",
                        help="where to save models and checkpoint")
    parser.add_argument("egs_dir", help="where feat_id_ark.list resides")

    return parser.parse_args()


class Distort(object):
    def __init__(self):
        kwargs = {
            "seed": 1111,
            "warp_width": 30,
            "max_freq_width": 5,
            "num_freq_masks": 2,
            "max_time_width": 50,
            "num_time_masks": 2,
            "max_workers": 30,
            "p": 0.2,
            "apply_time_warp": True,
            "inplace": True
        }

        self.spec_aug = SpecAugment(**kwargs)

    def __call__(self, args):
        feat, label = args
        distorted_feat = self.spec_aug(feat)
        yield distorted_feat, label


args = parse_args()
print(args)

model_dir = os.path.expandvars(args.model_dir)
seed = args.seed

device = torch.device('cuda:0')
torch.cuda.set_device(device)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

os.makedirs(model_dir, exist_ok=True)
model = build_model(args.conf,
                    model_type=args.model_type,
                    num_classes=args.num_classes,
                    write_back=True)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, dim=0)
model.to(device)

model_cfg = ConfigParser()
model_cfg.read(args.conf)
max_grad_value = model_cfg.getfloat(args.model_type, "max_grad_value")
max_grad_norm = model_cfg.getfloat(args.model_type, "max_grad_norm")

# prepare training data
batch_size = args.batch_size
min_chunk_size = args.min_chunk_size
max_chunk_size = args.max_chunk_size
train_list = os.path.join(args.egs_dir, "feat_id_ark.list")

tr_generator = KaldiXvector(data_list=train_list,
                            min_chunk_size=min_chunk_size,
                            max_chunk_size=max_chunk_size,
                            batch_size=batch_size,
                            in_memory=False,
                            blocks_per_load=40,
                            proportion=0.5,
                            max_workers=40,
                            shuffle=True,
                            seed=seed)
# preprocess_method=Distort())

stream = torch.cuda.Stream(device)


def to_device(args):
    feats, labels = args
    with torch.cuda.stream(stream):
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
    return feats, labels


train_data = DataQueue(tr_generator, max_queue_size=30)
train_loader = Prefetcher(train_data,
                          postprocess=to_device,
                          buffer_size=1,
                          stream=stream)

num_epochs = args.num_epochs
warmup_lr = args.warmup_lr
initial_lr = args.initial_lr
final_lr = args.final_lr
lr_anneal = (final_lr / initial_lr)**(1. / num_epochs)
lr_decline = (initial_lr - final_lr) / num_epochs
momentum = 0.9
nesterov = False
weight_decay = 1e-5

# Optimizer
optimizer = torch.optim.SGD(model.parameters(),
                            lr=initial_lr,
                            momentum=momentum,
                            nesterov=nesterov,
                            weight_decay=weight_decay)


# Set up learning rate scheduler
def get_learning_rate(epoch):
    """Compute learning rate of given epoch.

    Users can design different strategy to alter learning rate,
    Please make sure global variables like final_lr、
    lr_decline、initial_lr are assigned before.
    """
    if args.linear_decay:
        this_lr = max(final_lr, initial_lr - lr_decline * epoch)
    else:
        this_lr = max(final_lr, initial_lr * lr_anneal**epoch)

    if epoch == 0 and warmup_lr > 0:  # use warmup lr instead
        this_lr = warmup_lr
    return this_lr


lr_lambda = lambda epoch: get_learning_rate(epoch) / initial_lr
lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

log_dir = os.path.join(model_dir, f'log/dist{torch.cuda.device_count()}')
writer = SummaryWriter(log_dir)

# Train the Model
checkpoint_period = args.checkpoint_period
for epoch in range(args.start_epoch, num_epochs):

    if args.exit_epoch > 0 and args.exit_epoch == epoch:
        break

    ckpt_filename = os.path.join(model_dir, f'checkpoint_e{epoch-1:03d}.pkl')
    if epoch == args.start_epoch and os.path.isfile(ckpt_filename):
        ckpt = load_checkpoint(model, ckpt_filename, map_location='cpu')
        if args.start_epoch > 0:
            if isinstance(ckpt, dict) and 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
                print(f'load optimizer states from {ckpt_filename}')
            if isinstance(
                    ckpt,
                    dict) and 'meta' in ckpt and 'lr_state' in ckpt['meta']:
                lr_state = ckpt['meta'].get('lr_state')
                lr_scheduler.load_state_dict(lr_state)
                print(f'load scheduler states from {ckpt_filename}')
        print(f'load model from {ckpt_filename}')

    if epoch == args.start_epoch:
        nb_samples = epoch * args.frames_per_epoch

    writer.add_scalar('train/lr', lr_scheduler.get_lr()[0], nb_samples)

    total_loss = 0
    acc_samples = 0
    total_processed = 0
    steps = 0
    target_samples = checkpoint_period
    while total_processed < args.frames_per_epoch:
        inputs, labels = train_loader.get()
        inputs = inputs.to(device)
        labels = labels.to(device)

        b, t = inputs.size()[:2]

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        loss = model(inputs, labels).mean()
        loss.backward()

        if max_grad_value > 0:
            cur_max_value = max_grad_value
            clip_grad_value_(model.parameters(), clip_value=cur_max_value)
        if max_grad_norm > 0:
            cur_max_norm = max_grad_norm
            norm = clip_grad_norm_(model.parameters(), max_norm=cur_max_norm)
            if norm > cur_max_norm:
                print("grad norm {0:.2f} exceeds {1:.2f}, clip to {1:.2f}.".
                      format(norm, cur_max_norm))

        optimizer.step()

        loss_val = loss.item()
        del loss, inputs

        total_processed += b * t
        steps += 1
        nb_samples += b * t
        writer.add_scalar('train/loss', loss_val, nb_samples)
        total_loss += loss_val * b
        acc_samples += b
        if steps % 10 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %
                  (epoch + 1, num_epochs, total_processed,
                   args.frames_per_epoch, total_loss / acc_samples))
            total_loss = 0
            acc_samples = 0

        if checkpoint_period > 0 and total_processed >= target_samples:
            target_samples += checkpoint_period
            ckpt_filename = os.path.join(
                model_dir,
                'checkpoint_s{:06d}M.pkl'.format(nb_samples // 1000000))
            meta = {}
            meta['lr_state'] = lr_scheduler.state_dict()
            save_checkpoint(model,
                            ckpt_filename,
                            optimizer=optimizer,
                            meta=meta)

    lr_scheduler.step()
    ckpt_filename = os.path.join(
        model_dir, 'checkpoint_s{:06d}M.pkl'.format(nb_samples // 1000000))
    meta = {}
    meta['lr_state'] = lr_scheduler.state_dict()
    save_checkpoint(model, ckpt_filename, optimizer=optimizer, meta=meta)

    ckpt_linkname = os.path.join(model_dir,
                                 'checkpoint_e{:03d}.pkl'.format(epoch))
    cmd = "ln -sf ./checkpoint_s{:06d}M.pkl {}"
    cmd = cmd.format(nb_samples // 1000000, ckpt_linkname)
    subprocess.call(cmd, shell=True)
