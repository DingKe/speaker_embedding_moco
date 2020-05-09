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

from specaugment import SpecAugment
from pytorch.datasets.dataqueue import DataQueue, Prefetcher
from pytorch.datasets.kaldi_moco import KaldiMoCo
from pytorch.nnet.checkpoint import save_checkpoint, load_checkpoint
from pytorch.nnet.utils_moco import parse_splice_str, build_model, momentum_update, get_shuffle_ids
from pytorch.nnet.models_moco import MemoryMoCo


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="Launch Training ")

    # Optional args for training
    parser.add_argument("--beta",
                        type=float,
                        default=0.99,
                        help="momentum coefficent")
    parser.add_argument("--mem_queue_size",
                        type=int,
                        default=5000,
                        help="memory queue size")
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
    parser.add_argument("egs_dir", help="where feat_ark.list resides")

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
            "inplace": True
        }
        self.spec_aug = SpecAugment(**kwargs)

    def __call__(self, args):
        x1, x2 = args
        distorted_x1 = self.spec_aug(x1)
        distorted_x2 = self.spec_aug(x2)
        yield distorted_x1, distorted_x2


def get_data_loader(args):
    # prepare training data
    batch_size = args.batch_size
    min_chunk_size = args.min_chunk_size
    max_chunk_size = args.max_chunk_size
    train_list = None
    for type in ["", "id_", "pdf_", "disc_"]:
        file = os.path.join(args.egs_dir, f"feat_{type}ark.list")
        if os.path.isfile(file):
            train_list = file
            break
    assert train_list is not None, f"no feat_ark.list in {args.egs_dir}"

    tr_generator = KaldiMoCo(data_list=train_list,
                             min_chunk_size=min_chunk_size,
                             max_chunk_size=max_chunk_size,
                             batch_size=batch_size,
                             in_memory=False,
                             blocks_per_load=20,
                             proportion=0.5,
                             max_workers=40,
                             shuffle=True,
                             preprocess_method=Distort(),
                             seed=args.seed)

    return tr_generator


def train(args, data_loader):
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    stream = torch.cuda.Stream(device)

    def to_device(args):
        x1, x2 = args
        with torch.cuda.stream(stream):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
        return x1, x2

    train_data = DataQueue(data_loader, max_queue_size=30, nb_worker=10)
    train_loader = Prefetcher(train_data,
                              postprocess=to_device,
                              buffer_size=1,
                              stream=stream)

    model_cfg = ConfigParser()
    model_cfg.read(args.conf)
    max_grad_value = model_cfg.getfloat(args.model_type, "max_grad_value")
    max_grad_norm = model_cfg.getfloat(args.model_type, "max_grad_norm")

    beta = args.beta
    model_q = build_model(args.conf,
                          model_type=args.model_type,
                          write_back=True)
    model_k = build_model(args.conf,
                          model_type=args.model_type,
                          write_back=True)
    momentum_update(model_q, model_k, 1 - beta)

    embedding_size = model_cfg.getint(args.model_type, "embedding_size")
    memory = MemoryMoCo(embedding_size, args.mem_queue_size)

    if torch.cuda.device_count() > 1:
        model_q = nn.DataParallel(model_q, dim=0)
        model_k = nn.DataParallel(model_k, dim=0)
    model_q.to(device)
    model_k.to(device)
    memory.to(device)

    num_epochs = args.num_epochs
    warmup_lr = args.warmup_lr
    initial_lr = args.initial_lr
    final_lr = args.final_lr

    lr_anneal = (final_lr / initial_lr)**(1. / num_epochs)
    lr_decline = (initial_lr - final_lr) / num_epochs

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

    momentum = 0.9
    nesterov = False
    weight_decay = 1e-5

    # Optimizer
    optimizer = torch.optim.SGD(model_q.parameters(),
                                lr=initial_lr,
                                momentum=momentum,
                                nesterov=nesterov,
                                weight_decay=weight_decay)

    lr_lambda = lambda epoch: get_learning_rate(epoch) / initial_lr
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    model_dir = args.model_dir
    log_dir = os.path.join(model_dir, f'log/dist{torch.cuda.device_count()}')
    writer = SummaryWriter(log_dir)

    # Train the Model
    checkpoint_period = args.checkpoint_period
    for epoch in range(args.start_epoch, num_epochs):

        if args.exit_epoch > 0 and args.exit_epoch == epoch:
            break

        ckpt_filename = os.path.join(model_dir,
                                     f'checkpoint_e{epoch-1:03d}.pkl')
        if epoch == args.start_epoch and os.path.isfile(ckpt_filename):
            ckpt = load_checkpoint(model_q, ckpt_filename, map_location='cpu')
            if args.start_epoch > 0:
                if isinstance(ckpt, dict) and 'optimizer' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer'])
                    print(f'load optimizer states from {ckpt_filename}')
                if isinstance(
                        ckpt, dict
                ) and 'meta' in ckpt and 'lr_state' in ckpt['meta']:
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
            inputs, dis_inputs = train_loader.get()
            inputs = inputs.to(device)
            dis_inputs = dis_inputs.to(device)

            b, t = inputs.size()[:2]

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            with torch.no_grad():
                # Shuffle BN
                shf_ids, rev_ids = get_shuffle_ids(b, device)
                dis_inputs = dis_inputs[shf_ids]
                key = model_k(dis_inputs)[rev_ids].detach()
            query = model_q(inputs)

            loss = memory(query, key)
            loss.backward()

            if max_grad_value > 0:
                cur_max_value = max_grad_value
                clip_grad_value_(model_q.parameters(),
                                 clip_value=cur_max_value)
            if max_grad_norm > 0:
                cur_max_norm = max_grad_norm
                norm = clip_grad_norm_(model_q.parameters(),
                                       max_norm=cur_max_norm)
                if norm > cur_max_norm:
                    print(
                        "grad norm {0:.2f} exceeds {1:.2f}, clip to {1:.2f}.".
                        format(norm, cur_max_norm))

            optimizer.step()
            momentum_update(model_q, model_k, beta)
            memory.update(key)

            loss_val = loss.item()
            del loss, key, query, inputs, dis_inputs

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
                save_checkpoint(model_q,
                                ckpt_filename,
                                optimizer=optimizer,
                                meta=meta)

        lr_scheduler.step()
        ckpt_filename = os.path.join(
            model_dir, 'checkpoint_s{:06d}M.pkl'.format(nb_samples // 1000000))
        meta = {}
        meta['lr_state'] = lr_scheduler.state_dict()
        save_checkpoint(model_q, ckpt_filename, optimizer=optimizer, meta=meta)

        ckpt_linkname = os.path.join(model_dir,
                                     'checkpoint_e{:03d}.pkl'.format(epoch))
        cmd = "ln -sf ./checkpoint_s{:06d}M.pkl {}"
        cmd = cmd.format(nb_samples // 1000000, ckpt_linkname)
        subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    model_dir = os.path.expandvars(args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = model_dir

    data_loader = get_data_loader(args)

    train(args, data_loader)
