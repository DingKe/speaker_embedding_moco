#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
#             2020   Meituan-Dianping (Author: Ke Ding)
#             2020   Meituan-Dianping (Author: Xuanji He)
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.


root_dir=`pwd -P`
data=$root_dir/data
exp=$root_dir/exp
voxceleb1_trials=$data/test/trials
#dir=/opt/meituan/cephfs/user/hadoop-speech/dingke02/sid/archive/voxceleb/moco/exp/unsup_1a
dir=/opt/meituan/cephfs/user/hadoop-speech/hexuanji/sr/project/MoCo_Xvector/exp/xvector_1a
mdl=checkpoint_e050.pkl

plda_score=true
stage=0
nj=40

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;
set -e

umask 000

# Extract Xvectors
if [ $stage -le 1 ]; then
  # Extract Xvectors for centering, LDA, and PLDA training.
  local/extract_xvectors.sh \
    --cmd "$train_cmd" --nj 1000 \
    --mdl $mdl \
    $dir $data/train $exp/xvectors_train || exit 1;

  # Extract Xvectors used in the evaluation.
  local/extract_xvectors.sh \
    --cmd "$train_cmd" --nj $nj \
    --mdl $mdl \
    --chunk_size -1 \
    --pad-input true \
    $dir $data/test $exp/xvectors_test || exit 1;
fi

if [ $stage -le 2 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $run_cmd $exp/xvectors_train/log/compute_mean.log \
    ivector-mean scp:$exp/xvectors_train/xvector.scp \
    $exp/xvectors_train/mean.vec || exit 1;
fi

if [ $stage -le 3 ] && $plda_score; then
  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=150
  $run_cmd $exp/xvectors_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean $exp/xvectors_train/mean.vec scp:$exp/xvectors_train/xvector.scp ark:- |" \
    ark:$data/train/utt2spk $exp/xvectors_train/transform.mat || exit 1;

  # Train PLDA model.
  $run_cmd $exp/xvectors_train/log/plda.log \
    ivector-compute-plda ark:$data/train/spk2utt \
    "ark:ivector-subtract-global-mean $exp/xvectors_train/mean.vec scp:$exp/xvectors_train/xvector.scp ark:- | transform-vec $exp/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $exp/xvectors_train/plda || exit 1;
fi

trials_out_plda=$exp/trials_out_plda
if [ $stage -le 4 ] && $plda_score; then
  $run_cmd $exp/xvectors_test/log/plda_score.log \
    ivector-plda-scoring --normalize-length=true \
      "ivector-copy-plda --smoothing=0.0 $exp/xvectors_train/plda - |" \
      "ark:ivector-subtract-global-mean $exp/xvectors_train/mean.vec scp:$exp/xvectors_test/xvector.scp ark:- | transform-vec $exp/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-subtract-global-mean $exp/xvectors_train/mean.vec scp:$exp/xvectors_test/xvector.scp ark:- | transform-vec $exp/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" $trials_out_plda || exit 1;
fi

trials_out_cosine=$exp/trials_out_cosine
if [ $stage -le 5 ] && ! $plda_score; then
  $run_cmd $exp/xvectors_test/log/cosine_score.log \
    local/cosine_scoring.py \
      $exp/xvectors_test/xvector.scp \
      $exp/xvectors_test/xvector.scp \
      $exp/xvectors_train/mean.vec \
      $voxceleb1_trials $trials_out_cosine || exit 1;
fi

if [ $stage -le 6 ]; then
  if $plda_score; then trials_out=$trials_out_plda; else trials_out=$trials_out_cosine; fi
  # eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials $trials_out) 2> /dev/null`
  local/prepare_for_eer.py $voxceleb1_trials $trials_out | compute-eer - || exit 1;
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $trials_out $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $trials_out $voxceleb1_trials 2> /dev/null`
  # echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi


