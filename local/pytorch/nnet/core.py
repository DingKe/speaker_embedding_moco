# -*- coding: utf-8 -*-
from __future__ import absolute_import

import collections
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


def get_activation(callable_or_str):
    if callable_or_str is None:
        return lambda x: x
    elif callable(callable_or_str):
        return callable_or_str
    else:
        return getattr(F, callable_or_str.lower())


def np2tensor(arr):
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr)
    elif isinstance(arr, collections.Sequence):
        return [np2tensor(a) for a in arr]
    elif isinstance(arr, collections.Mapping):
        return {key: np2tensor(val) for key, val in arr.items()}
    else:
        return arr


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        is_packed = isinstance(x, PackedSequence)

        if is_packed:
            x, batch_sizes = x.data, x.batch_sizes

        if len(x.size()) > 2:
            leading = x.size()[:2]
            tailing = x.size()[2:]

            x = x.view((-1, ) + tailing)
            x = self.module(x)

            new_size = leading + x.size()[1:]
            x = x.view(new_size)
        else:
            x = self.module(x)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

        def reset_parameters(self):
            if hasattr(self.module, "reset_parameters"):
                self.module.reset_parameters()

        def __repr__(self):
            tmpstr = self.__class__.__name__ + ' (\n'
            tmpstr += self.module.__repr__()
            tmpstr += ')'
            return tmpstr


class Splice(nn.Module):
    def __init__(self,
                 context,
                 full_context=False,
                 batch_first=True,
                 keep_dims=False):
        """
        Splice multiple frames.
        Helpful for implementing Time Delayed Neural Network (TDNN).
        :param context:
        """
        super(Splice, self).__init__()

        self.check_valid_context(context)
        self.context = self.normalize_context(context, full_context)
        self.batch_first = batch_first
        self.keep_dims = keep_dims

    def forward(self, x):
        assert len(x.size()) == 3, "Input should be a 3D tensor"

        input_size = x.size()
        if self.batch_first:
            [batch_size, input_sequence_length, input_dim] = input_size
        else:
            [input_sequence_length, batch_size, input_dim] = input_size

        # Allocate memory for output
        start, end = self.get_valid_steps(self.context, input_sequence_length)
        output_sequence_length = end - start
        if self.batch_first:
            xs = x.new_empty(batch_size, output_sequence_length,
                             len(self.context), input_dim)
        else:
            xs = x.new_empty(output_sequence_length, batch_size,
                             len(self.context), input_dim)

        for i, offset in enumerate(self.context):
            if self.batch_first:
                xs[:, :, i, :] = x[:, start + offset:end + offset, :]
            else:
                xs[:, :, i, :] = x[start + offset:end + offset, :, :]

        if self.keep_dims:
            return xs

        if self.batch_first:
            return xs.view((batch_size, output_sequence_length,
                            len(self.context) * input_dim))
        else:
            return xs.view((output_sequence_length, batch_size,
                            len(self.context) * input_dim))

    @staticmethod
    def check_valid_context(context):
        # here context is still a list
        assert context[0] <= context[-1], 'Invalid context'

    @staticmethod
    def normalize_context(context, full_context):
        if full_context:
            context = list(range(context[0], context[-1] + 1))
        return context

    @staticmethod
    def get_valid_steps(context, input_sequence_length):
        start = 0 if context[0] >= 0 else -context[0]
        end = input_sequence_length if context[
            -1] <= 0 else input_sequence_length - context[-1]
        return start, end


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, normalize,
                 activation):
        super(MLP, self).__init__()

        self.activation = get_activation(activation)
        '''
        if not isinstance(normalize, nn.Module):
            normalize = getattr(nn, normalize)
        '''
        self.l1 = nn.Linear(input_size, hidden_size)
        self.n1 = normalize(hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        self.n2 = normalize(output_size)

        self.reset_parameters()

    def forward(self, input):
        x = self.n1(self.activation(self.l1(input)))
        x = self.n2(self.activation(self.l2(x)))
        return x

    def reset_parameters(self):
        for child in self.children():
            child.reset_parameters()


class StatsPooling(nn.Module):
    def __init__(self, eps=1e-5, input_size=None):
        super(StatsPooling, self).__init__()
        self.eps = eps
        self.input_size = input_size

    def forward(self, input):
        mean = input.mean(dim=1)
        var = input.var(dim=1)
        std = torch.sqrt(var + self.eps)
        return torch.cat((mean, std), dim=1)


class TDNN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 context,
                 full_context=False,
                 activation='relu',
                 bias=True):
        """
        :param input_dim:
        """
        super(TDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context = context
        self.ful_context = full_context
        self.activation = get_activation(activation)
        self.bias = bias

        self.splice = Splice(context, full_context)
        linear = nn.Linear(input_dim * len(self.splice.context), output_dim,
                           bias)
        self.linear = SequenceWise(linear)

    def forward(self, x):
        y = self.linear(self.splice(x))
        y = self.activation(y)
        return y


class AAMLoss(nn.Module):
    """Additive Arngular Marngin Loss

    Reference:
    Deng et al. ArcFace: Additive Angular Margin Loss for Deep Face Recognition

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self,
                 num_classes,
                 feat_dim,
                 margin_s=32.0,
                 margin_m=0.30,
                 easy_margin=False,
                 eps=1e-5):
        super(AAMLoss, self).__init__()

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.s = margin_s
        self.m = margin_m
        self.easy_margin = easy_margin
        self.eps = eps

        # precompute constants.
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        self.criterion = nn.CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, targets):
        x = nn.functional.normalize(features, p=2, dim=1)
        W = nn.functional.normalize(self.weight, p=2, dim=1)

        cosine = nn.functional.linear(x, W)
        cosine.clamp_(-1 + self.eps, 1 - self.eps)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s

        loss = self.criterion(logits, targets)

        return loss
