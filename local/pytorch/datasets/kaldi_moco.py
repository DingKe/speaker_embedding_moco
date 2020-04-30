# -*- coding: utf-8 -*-
from __future__ import absolute_import

import concurrent
import concurrent.futures

import os
import sys
import numpy as np
from bisect import bisect_left

from .generator import BaseGenerator
from .dataqueue import BackgroundGenerator
import kaldi_io


class KaldiMoCo(BaseGenerator):
    """Data loader for MoCo like training.
    """
    def __init__(self,
                 data_list,
                 min_chunk_size,
                 max_chunk_size,
                 in_memory=False,
                 blocks_per_load=40,
                 proportion=0.5,
                 max_workers=3,
                 **kwargs):
        '''
        Args:
          data_list: each line with two fields: feat.ark utt2int
          min_chunk_size, max_chunk_size: sampled utts will be truncated
            between [min_chunk_size, max_chunk_size].
            Caution: max_chunk_size must <= the minimum frame numeber
            of the whole dataset.
          in_memory: if true, load the whole dataset in memory.
          blocks_per_load: if not in_memory, load this many arks at one time.
            Ignored if in_memory.
          proportion: for each load, feed #total frames * proportion frames.
            Ingored if in_memory.
        '''

        super(KaldiMoCo, self).__init__(**kwargs)
        del kwargs

        self.__dict__.update(locals())
        self.__dict__.pop('self')

        self.data_list = os.path.expandvars(self.data_list)

    def _preload(self, block_list):
        for i in range(0, len(block_list), self.blocks_per_load):
            blocks = block_list[i:i + self.blocks_per_load]
            sentences = self._load_sentences(blocks)

            num_frames = [len(frames) for _, frames in sentences]
            cdf = np.cumsum(num_frames)
            target_frames = int(self.proportion * cdf[-1])
            cdf = cdf / cdf[-1]

            yield sentences, cdf, target_frames

    def __call__(self):
        block_list = self._load_block_list(self.data_list)

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers)

        if self.shuffle:
            np.random.seed(self.seed)
        else:
            self.proportion = 1

        if self.in_memory:
            # load the whole dataset into memory
            self.blocks_per_load = len(block_list)

        while True:
            if self.shuffle:
                self.random.shuffle(block_list)

            sentence_gen = BackgroundGenerator(self._preload(block_list))
            for sentences, cdf, target_frames in sentence_gen:
                batch_size = min(self.batch_size, len(sentences))
                total_steps = len(sentences) // batch_size
                step = 0
                count = 0
                while count < target_frames:
                    # feed data one mini-batch each time
                    sub_sentences = []
                    if self.shuffle:
                        rns = np.random.rand(batch_size)
                    else:
                        if step == total_steps:
                            break
                        rns = cdf[step * batch_size:(step + 1) * batch_size]
                        step += 1
                    for r in rns:
                        idx = bisect_left(cdf, r)
                        sub_sentences.append(sentences[idx])

                    chunked_sentences_1 = self._chunk(sub_sentences)
                    chunked_sentences_2 = self._chunk(sub_sentences)

                    x1 = np.asarray([feat for _, feat in chunked_sentences_1],
                                    dtype='float32')
                    x2 = np.asarray([feat for _, feat in chunked_sentences_2],
                                    dtype='float32')

                    if not self.in_memory:
                        count += batch_size * (x1.shape[1] + x2.shape[1]) // 2

                    yield x1, x2

                del sentences

    def _load_block_list(self, data_list):
        block_list = []
        with open(data_list, 'r') as fptr:
            # eachline: feat_ark ...
            for line in fptr:
                feat_ark = line.strip().split()[0]
                block_list.append(feat_ark)

        return block_list

    def _load_sentences(self, blocks):
        # don't bother using concurrency
        if len(blocks) == 1:
            return self._load_block(blocks[0])

        sentences = []
        futures = []
        for block in blocks:
            future = self.executor.submit(self._load_block, block)
            futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            sentences.extend(future.result())
        return sentences

    def _load_block(self, feat_ark):
        sentences = []

        if self.verbose >= 1:
            print(f"[Loading] feat ark: {feat_ark}", file=sys.stderr)

        rxfilename = f"copy-feats ark:{feat_ark} ark:- |"
        feat_gen = kaldi_io.read_mat_ark(rxfilename)
        for key, frames in feat_gen:
            # fake labels to simplify implementation
            new_sentence = (key, frames)
            sentences.append(new_sentence)

        if self.verbose >= 1:
            print("[Done] feat ark: {}. {} sentences loaded".format(
                feat_ark, len(sentences)))

        return sentences

    def _chunk(self, sentences):
        if self.shuffle:
            chunk_size = self.random.randint(self.min_chunk_size,
                                             self.max_chunk_size)
        else:
            chunk_size = self.max_chunk_size
        chunked_sentences = []
        for sentence in sentences:
            key, frames = sentence
            if self.shuffle:
                offset = self.random.randint(0, len(frames) - chunk_size)
            else:
                offset = 0
            frames = frames[offset:offset + chunk_size]
            chunked_sentences.append((key, frames))
        return chunked_sentences
