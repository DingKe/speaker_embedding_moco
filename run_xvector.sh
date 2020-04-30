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


# The trials file is downloaded by local/make_voxceleb1.pl.
voxceleb1_root=/opt/meituan/cephfs/user/hadoop-speech/sid_datasets/voxceleb/voxceleb1
voxceleb2_root=/opt/meituan/cephfs/user/hadoop-speech/sid_datasets/voxceleb/voxceleb2
rirs_noises_root=/opt/meituan/cephfs/user/hadoop-speech/noise_rir/slr28/RIRS_NOISES
musan_root=/opt/meituan/cephfs/user/hadoop-speech/noise_rir/musan/musan
moco_model=

root_dir=`pwd -P`
data=$root_dir/Voxceleb/xvector/data
exp=$root_dir/Voxceleb/xvector/exp
dir=$exp/xvector_1a
voxceleb1_trials=$data/test/trials

do_augment=true
min_len=500
min_num_utts=8
rate=16000 # sample rate
stage=0
nj=400

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;
set -e

umask 000

[ ! -d RIRS_NOISES ] && ln -sf $rirs_noises_root

if [ $stage -le 0 ]; then
  local/make_voxceleb1_v2.pl $voxceleb1_root dev $data/voxceleb1_train
  local/make_voxceleb1_v2.pl $voxceleb1_root test $data/voxceleb1_test
  local/make_voxceleb2.pl $voxceleb2_root dev $data/voxceleb2_train
  local/make_voxceleb2.pl $voxceleb2_root test $data/voxceleb2_test
  # We'll train on all of VoxCeleb2, plus the training portion of VoxCeleb1.
  # This should give 7,351 speakers and 1,277,503 utterances.
  utils/combine_data.sh $data/train $data/voxceleb2_train $data/voxceleb2_test $data/voxceleb1_train || exit 1;
  ln -sf $data/voxceleb1_test $data/test
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in train test; do
    run_mfcc_nj=40
    # if [ $name == 'test' ]; then run_mfcc_nj=40; fi
    mfccdir=$data/$name/mfcc
    vaddir=$mfccdir
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $run_mfcc_nj --cmd "$run_cmd" \
      $data/${name} $exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh $data/${name}
    sid/compute_vad_decision.sh --nj $run_mfcc_nj --cmd "$run_cmd" \
      $data/${name} $exp/make_vad $vaddir
    utils/fix_data_dir.sh $data/${name}
  done
fi

# In this section, we augment the VoxCeleb2 data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ $stage -le 2 ] && $do_augment; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' $data/train/utt2num_frames > $data/train/reco2dur


  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, $rirs_noises_root/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, $rirs_noises_root/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
  # additive noise here.
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate $rate \
    $data/train $data/train_reverb
  cp $data/train/vad.scp $data/train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" $data/train_reverb $data/train_reverb.new
  rm -rf $data/train_reverb
  mv $data/train_reverb.new $data/train_reverb
  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh $musan_root $data || exit 1;

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh $data/musan_${name}
    mv $data/musan_${name}/utt2dur $data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "$data/musan_noise" $data/train $data/train_noise
  # Augment with musan_music
  python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "$data/musan_music" $data/train $data/train_music
  # Augment with musan_speech
  python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "$data/musan_speech" $data/train $data/train_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh $data/train_aug $data/train_reverb $data/train_noise $data/train_music $data/train_babble
fi

if [ $stage -le 3 ] && $do_augment; then
  # Take a random subset of the augmentations
  utils/subset_data_dir.sh $data/train_aug 1000000 $data/train_aug_1m
  utils/fix_data_dir.sh $data/train_aug_1m

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$run_cmd" \
    $data/train_aug_1m $exp/make_mfcc $mfccdir

  # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh $data/train_combined $data/train_aug_1m $data/train
else
  [ ! -d $data/train_combined ] && ln -sf $data/train $data/train_combined
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/prepare_feats_for_egs.sh --nj $nj --cmd "$train_cmd" \
    $data/train_combined $data/train_combined_no_sil $exp/train_combined_no_sil
  utils/fix_data_dir.sh $data/train_combined_no_sil
fi

if [ $stage -le 5 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  mv $data/train_combined_no_sil/utt2num_frames $data/train_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $data/train_combined_no_sil/utt2num_frames.bak > $data/train_combined_no_sil/utt2num_frames
  utils/filter_scp.pl $data/train_combined_no_sil/utt2num_frames $data/train_combined_no_sil/utt2spk > $data/train_combined_no_sil/utt2spk.new
  mv $data/train_combined_no_sil/utt2spk.new $data/train_combined_no_sil/utt2spk
  utils/fix_data_dir.sh $data/train_combined_no_sil
  
  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  awk '{print $1, NF-1}' $data/train_combined_no_sil/spk2utt > $data/train_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $data/train_combined_no_sil/spk2num | utils/filter_scp.pl - $data/train_combined_no_sil/spk2utt > $data/train_combined_no_sil/spk2utt.new || exit 1;
  mv $data/train_combined_no_sil/spk2utt.new $data/train_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl $data/train_combined_no_sil/spk2utt > $data/train_combined_no_sil/utt2spk

  utils/filter_scp.pl $data/train_combined_no_sil/utt2spk $data/train_combined_no_sil/utt2num_frames > $data/train_combined_no_sil/utt2num_frames.new
  mv $data/train_combined_no_sil/utt2num_frames.new $data/train_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh $data/train_combined_no_sil
fi

# Stages 6 through 8 are handled in run_xvector.sh
[ -f $moco_model ] && mkdir -p $dir && ln -sf $moco_model $dir/checkpoint_e-01.pkl
local/run_xvector.sh \
  --stage $stage \
  --num_repeats 50 \
  --start_epoch 0 \
  --batch_size 1024 \
  --min_chunk_size 200 \
  --max_chunk_size 400 \
  --initial_lr_per_sample 1e-5 \
  --final_lr_per_sample 1e-6 \
  --cmd "$train_cmd" --nj $nj \
  --data $data/train_combined_no_sil \
  --dir $dir --egs-dir $dir/egs || exit 1;

exit 0;
