#!/usr/bin/env python3

import sys
import numpy as np
import argparse
from sklearn import preprocessing
import kaldi_io

parser = argparse.ArgumentParser()
parser.add_argument('--impostor_scp',
                    type=str,
                    default='',
                    help='impostor data path')
parser.add_argument('--top_percent',
                    type=float,
                    default='0.1',
                    help='top percent of asnorm')
parser.add_argument('--eps', type=float, default='1e-6', help='eps')
parser.add_argument('enroll_scp')
parser.add_argument('eval_scp')
parser.add_argument('mean_file')
parser.add_argument('trials')
parser.add_argument('trials_out')
args = parser.parse_args()


def read_target_vector(file_path, targets):
    ''' read from vectors in targets
    '''
    spks = []
    feats = []
    feat_gen = kaldi_io.read_vec_flt_scp(file_path)
    for key, feat in feat_gen:
        if key in targets:
            spks.append(key)
            feats.append(feat)
    return spks, feats


def read_impostor_vector(file_path):
    '''read impostor vector from scp
    '''
    feats = []
    feat_gen = kaldi_io.read_vec_flt_scp(file_path)
    for key, feat in feat_gen:
        feats.append(feat)
    return feats


def compute_mean_and_std(result, top_percent):
    total_num = len(result)
    top_size = round(total_num * top_percent)
    result = -np.sort(-result)[:top_size]
    mean = np.mean(result)
    std = np.std(result)
    return mean, std


def zscore_normalization(x, mean, std):
    x = (x - mean) / (std + args.eps)
    return x


trial_utts = set()
trial_spks = set()
with open(args.trials, 'r') as fptr:
    for line in fptr:
        enroll_spk, eval_utt, target = line.strip().split()
        trial_utts.add(eval_utt)
        trial_spks.add(enroll_spk)

# Read mean, enroll embedding and eval embedding
mean = kaldi_io.read_vec_flt(args.mean_file)
enroll_spks, enroll_feats = read_target_vector(args.enroll_scp, trial_spks)
eval_utts, eval_feats = read_target_vector(args.eval_scp, trial_utts)
if args.impostor_scp != '':
    impostor_feats = read_impostor_vector(args.impostor_scp)
    impostor_feats = np.array(impostor_feats, dtype=np.float32)
    impostor_feats = impostor_feats - mean
    impostor_feats = preprocessing.normalize(impostor_feats, norm='l2')
# Convert data to numpy
enroll_feats = np.array(enroll_feats, dtype=np.float32)
eval_feats = np.array(eval_feats, dtype=np.float32)
enroll_spks = np.array(enroll_spks)
eval_utts = np.array(eval_utts)

# Subtract mean for enroll and eval  embedding
enroll_feats = enroll_feats - mean
eval_feats = eval_feats - mean

# Length normalize for enroll and eval embedding
enroll_feats = preprocessing.normalize(enroll_feats, norm='l2')
eval_feats = preprocessing.normalize(eval_feats, norm='l2')

# Compute cosine distance
cos_res = eval_feats.dot(enroll_feats.transpose())
if args.impostor_scp != '':
    eval_impostor_rs = eval_feats.dot(impostor_feats.transpose())
    enroll_impostor_rs = enroll_feats.dot(impostor_feats.transpose())

num_trials_err = 0
num_trials_done = 0
with open(args.trials, 'r') as fptr, \
     open(args.trials_out, 'w') as fout:
    for line in fptr:
        enroll_spk, eval_utt, target = line.strip().split()

        pos = np.where(eval_utts == eval_utt)[0]
        if len(pos) == 0:
            print(f"{eval_utt} not present in eval vectors.", file=sys.stderr)
            num_trials_err += 1
            continue
        else:
            eval_pos = pos[0]

        pos = np.where(enroll_spks == enroll_spk)[0]
        if len(pos) == 0:
            print(f"{enroll_spk} not present in enroll vectors.",
                  file=sys.stderr)
            num_trials_err += 1
            continue
        else:
            enroll_pos = pos[0]

        score = cos_res[eval_pos, enroll_pos]
        if args.impostor_scp != '':
            enroll_mean, enroll_std = compute_mean_and_std(
                enroll_impostor_rs[enroll_pos], args.top_percent)
            eval_mean, eval_std = compute_mean_and_std(
                eval_impostor_rs[eval_pos], args.top_percent)
            score = zscore_normalization(score, enroll_mean,
                                         enroll_std) + zscore_normalization(
                                             score, eval_mean, eval_std)

        fout.write(f"{enroll_spk} {eval_utt} {score}\n")
        num_trials_done += 1

print(f"Processed {num_trials_done} trials, {num_trials_err} had error.",
      file=sys.stderr)
