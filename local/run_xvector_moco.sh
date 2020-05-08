#!/usr/bin/bash

stage=6
nj=40
cmd=run.pl

dir=exp/moco

data=data/train_no_sil
egs_dir=

# start training config
num_repeats=30
frames_per_epoch=72000000  # 200 hours
start_epoch=0
exit_epoch=-1
checkpoint_period=0

utt_per_ark=2000
min_chunk_size=200
max_chunk_size=500
mem_queue_size=10000

warmup_lr_per_sample=0
initial_lr_per_sample=1e-4
final_lr_per_sample=1e-5
batch_size=1024  # 256 for each GPU, each consumes ~13G GPU mem
# end training config

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh

[ -z "$egs_dir" ] && egs_dir=$dir/egs

mkdir -p $dir $egs_dir || exit 1;
egs_dir=$(utils/make_absolute.sh $egs_dir)

required="utt2num_frames feats.scp"
for f in $required; do
 [ ! -f $data/$f ] && echo "$data/$f not exists!" && exit 1;
done

if [ $stage -le 6 ]; then
  cp $data/utt2num_frames $egs_dir
fi

if [ $stage -le 7 ]; then
  nutter=`cat $data/feats.scp | wc -l`
  total_nj=$((($nutter + $utt_per_ark - 1) / $utt_per_ark ))

  mkdir -p $egs_dir/tmp || exit 1;
  trap "rm -rf $egs_dir/tmp" INT HUP EXIT TERM QUIT

  cat $data/feats.scp | utils/shuffle_list.pl --srand 1111 >$egs_dir/tmp/feats.scp
  split_scps=
  for n in $(seq $total_nj); do
    split_scps="$split_scps $egs_dir/tmp/feats.scp.$n"
  done

  utils/split_scp.pl $egs_dir/tmp/feats.scp $split_scps || exit 1;

  $cmd --max-jobs-run $nj JOB=1:$total_nj $egs_dir/tmp/copy_feats.JOB.log \
    copy-feats --compress=true \
    scp:$egs_dir/tmp/feats.scp.JOB ark,scp:$egs_dir/feats.JOB.ark,$egs_dir/feats.JOB.scp || exit 1;

  for i in `seq $total_nj`; do
    echo "$egs_dir/feats.$i.ark"
  done > $egs_dir/feat_ark.list
fi

if [ $stage -le 8 ]; then
  num_frames=$(awk '{n += $2} END {print n}' <$egs_dir/utt2num_frames)
  num_epochs=$[($num_frames * $num_repeats) / $frames_per_epoch + 1]

  echo $num_frames >$egs_dir/num_frames
  echo $frames_per_epoch >$egs_dir/frames_per_epoch

  warmup_lr=$(perl -e "printf '%.2e', $warmup_lr_per_sample * $batch_size")
  initial_lr=$(perl -e "printf '%.2e', $initial_lr_per_sample * $batch_size")
  final_lr=$(perl -e "printf '%.2e', $final_lr_per_sample * $batch_size")

  echo $min_chunk_size >$dir/min_chunk_size
  echo $max_chunk_size >$dir/max_chunk_size

  ./local/train_moco.py --seed 0611 \
    --conf "conf/model.conf" \
    --mem_queue_size $mem_queue_size \
    --batch_size $batch_size \
    --frames_per_epoch $frames_per_epoch \
    --warmup_lr $warmup_lr --initial_lr $initial_lr --final_lr $final_lr \
    --min_chunk_size $min_chunk_size --max_chunk_size $max_chunk_size \
    --checkpoint_period $checkpoint_period \
    --num_epochs $num_epochs --start_epoch $start_epoch --exit_epoch $exit_epoch \
    $dir $egs_dir

  ln -sf ./checkpoint_e$[num_epochs-1].pkl $dir/final.pkl
fi
