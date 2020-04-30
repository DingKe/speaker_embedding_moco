# -*- coding: utf-8 -*-
from __future__ import absolute_import

import concurrent
import concurrent.futures

import os
import sys
import numpy as np
from bisect import bisect_left

from .generator import BaseGenerator
import kaldi_io


class KaldiXvector(BaseGenerator):
    """Data loader for xvector training.
    """
    def __init__(self,
                 data_list,
                 min_chunk_size,
                 max_chunk_size,
                 in_memory=False,
                 blocks_per_load=40,
                 proportion=0.5,
                 inference=False,
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
          inference: if true, no utt2int is needed, a fake id is provided.
        '''

        super(KaldiXvector, self).__init__(**kwargs)
        del kwargs

        self.__dict__.update(locals())
        self.__dict__.pop('self')

        self.data_list = os.path.expandvars(self.data_list)

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

            for i in range(0, len(block_list), self.blocks_per_load):
                blocks = block_list[i:i + self.blocks_per_load]
                sentences = self._load_sentences(blocks)

                num_frames = [len(frames) for _, frames, _ in sentences]
                cdf = np.cumsum(num_frames)
                cdf = cdf / cdf[-1]

                target_frames = int(self.proportion * sum(num_frames))
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

                    chunked_sentences = self._chunk(sub_sentences)

                    feats = np.asarray(
                        [feat for _, feat, _ in chunked_sentences],
                        dtype='float32')
                    labels = np.asarray(
                        [idx for _, _, idx in chunked_sentences],
                        dtype='int64')

                    if not self.in_memory:
                        count += batch_size * feats.shape[1]

                    yield feats, labels

                del sentences

    def _load_block_list(self, data_list):
        block_list = []
        with open(data_list, 'r') as fptr:
            # eachline: feat_ark utt2int
            for line in fptr:
                if not self.inference:
                    feat_ark, utt2int = line.strip().split()[:2]
                else:
                    feat_ark, utt2int = line.strip().split()[0], None
                block_list.append((feat_ark, utt2int))

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

    def _load_block(self, block):
        feat_ark, utt2int = block
        sentences = []

        if self.verbose >= 1:
            print(f"[Loading] feat ark: {feat_ark}", file=sys.stderr)

        rxfilename = f"copy-feats ark:{feat_ark} ark:- |"
        feat_gen = kaldi_io.read_mat_ark(rxfilename)
        if self.inference:
            for key, frames in feat_gen:
                # fake labels to simplify implementation
                new_sentence = (key, frames, 0)
                sentences.append(new_sentence)
        else:
            id_map = {}
            with open(utt2int, "r") as fptr:
                for line in fptr:
                    key, idx = line.strip().split()
                    id_map[key] = int(idx)

            for key, frames in feat_gen:
                if key not in id_map: continue

                new_sentence = (key, frames, id_map[key])
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
            key, frames, idx = sentence
            if self.shuffle:
                offset = self.random.randint(0, len(frames) - chunk_size)
            else:
                offset = 0
            frames = frames[offset:offset + chunk_size]
            chunked_sentences.append((key, frames, idx))
        return chunked_sentences
