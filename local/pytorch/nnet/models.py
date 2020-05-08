# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import Splice, get_activation, MLP, StatsPooling, TDNN, SequenceWise, AAMLoss


class XvectorNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        embedding_size,
        num_classes,
        splice_indexes,  # "-2,-1,0,1,2 -2,0,2 -3,0,3"
        activation="relu",
        normalize="BatchNorm1d",
        p=0,
        metric='softmax',
        mode="train"):
        super(XvectorNet, self).__init__()

        self.splice_indexes = splice_indexes
        self.metric = metric

        assert metric in ["softmax", "aam"], f"Unsupported metric {metric}"

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

        if metric == 'softmax':
            self.norm_b = self.normalize(embedding_size)
            # final affine layer without softmax
            self.final = nn.Linear(embedding_size, num_classes)
            self.criterion = nn.CrossEntropyLoss()
        elif metric == 'aam':
            self.criterion = AAMLoss(num_classes, embedding_size)

    def forward(self, x, labels):

        encoded = self.encode(x)

        embed_a = self.embed_a(encoded)
        post_a = self.norm_a(self.activation(embed_a))

        embed_b = self.embed_b(self.p(post_a))

        if self.metric == "softmax":
            x = self.norm_b(self.activation(embed_b))
            logit = self.final(x)
        elif self.metric == "aam":
            logit = embed_b

        return self.criterion(logit, labels)

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
        # delete final layer for embedding only.
        if self.metric == "softmax":
            del self.final, self.norm_b

        del self.criterion

        if embed_a_only:
            del self.norm_a, self.embed_b

    @property
    def extra_context(self):
        left, right = 0, 0
        for context in self.splice_indexes:
            left += -min(0, context[0])
            right += max(0, context[-1])
        return left, right
