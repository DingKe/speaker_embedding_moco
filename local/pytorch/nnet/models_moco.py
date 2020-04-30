# -*- coding: utf-8 -*-

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import Splice, get_activation, MLP, StatsPooling, TDNN, SequenceWise, AAMLoss


class MemoryMoCo(nn.Module):
    def __init__(self, feat_dim, K, T=0.07):
        '''
        K: memory queue size
        T: temperature
        '''
        super(MemoryMoCo, self).__init__()
        self.feat_dim = feat_dim
        self.queue_size = K
        self.T = T
        self.index = 0

        self.criterion = nn.CrossEntropyLoss()
        self.labels = None

        memory = torch.empty(K, feat_dim)
        nn.init.kaiming_uniform_(memory, a=1)
        self.register_buffer("memory", memory)

    def update(self, feats, offset=0, stride=None):
        batch_size = feats.size(0)
        self.index = (self.index + offset) % self.queue_size
        with torch.no_grad():
            out_ids = torch.arange(batch_size)
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queue_size)
            out_ids = out_ids.long().to(self.memory.device)
            self.memory.index_copy_(0, out_ids, feats)

        stride = batch_size if stride is None else stride
        self.index = (self.index + stride) % self.queue_size
        self.labels = None

    def forward(self, q, k, update=False):
        batch_size = q.size(0)
        k = k.detach()

        # positive logits: batch_size x 1
        l_pos = torch.bmm(q.view(batch_size, 1, -1), k.view(batch_size, -1, 1))
        l_pos = l_pos.view(batch_size, 1)

        # negative logits: batch_size x K
        queue = self.memory.detach().to(q)
        l_neg = F.linear(q, queue)

        # logits: batch_size x (1 + K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T

        # label for CE. 0s for postive logits
        if self.labels is None or self.labels.numel() != batch_size:
            self.labels = torch.zeros(batch_size, dtype=torch.long)
        self.labels = self.labels.to(logits.device)

        loss = self.criterion(logits, self.labels)

        if update:
            self.update(k)

        return loss


class XvectorNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        embedding_size,
        splice_indexes,  # "-2,-1,0,1,2 -2,0,2 -3,0,3"
        activation="relu",
        normalize="BatchNorm1d",
        p=0,
        mode="train"):
        super(XvectorNet, self).__init__()

        self.splice_indexes = splice_indexes

        try:
            self.normalize = getattr(nn, normalize)
        except:
            print("not found {} from nn, use LayerNorm".format(normalize))
            self.normalize = nn.BatchNorm1d
        self.activation = get_activation(activation)

        self.mode = mode
        assert self.mode in ["train", "test"], "Invalid mode"

        module_list = []

        # TDNN layers
        for l, context in enumerate(splice_indexes):
            if l == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
            output_dim = hidden_size
            tdnn = TDNN(input_dim, output_dim, context, activation=activation)
            module_list.extend(
                [tdnn, SequenceWise(self.normalize(output_dim))])

        # linear layers before pooling
        pre_stat = MLP(hidden_size,
                       hidden_size,
                       output_size,
                       normalize=self.normalize,
                       activation=self.activation)
        module_list.append(SequenceWise(pre_stat))

        # pooling layer
        module_list.append(StatsPooling())

        self.encode = nn.Sequential(*module_list)

        # embeding layers
        self.embed_a = nn.Linear(output_size * 2, hidden_size)
        self.norm_a = self.normalize(hidden_size)
        self.p = nn.Dropout(p)
        self.embed_b = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):

        encoded = self.encode(x)

        embed_a = self.embed_a(encoded)
        post_a = self.norm_a(self.activation(embed_a))

        embed_b = self.embed_b(self.p(post_a))
        logit = F.normalize(embed_b)

        return logit

    def extract_xvector(self, x):
        return self.extract_xvector_a(x)

    def extract_xvector_a(self, x):
        encoded = self.encode(x)
        embed_a = self.embed_a(encoded)
        return embed_a

    def extract_xvector_all(self, x):
        encoded = self.encode(x)
        embed_a = self.embed_a(encoded)
        post_a = self.norm_a(self.activation(embed_a))
        embed_b = self.embed_b(self.p(post_a))
        return embed_a, embed_b

    def prune(self, embed_a_only=False):
        if embed_a_only:
            del self.norm_a, self.embed_b

    @property
    def extra_context(self):
        left, right = 0, 0
        for context in self.splice_indexes:
            left += -min(0, context[0])
            right += max(0, context[-1])
        return left, right
