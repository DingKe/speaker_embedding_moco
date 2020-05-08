export run_cmd="${AM_ROOT}/kaldi_utils/parallel/run.pl"
export spark_cmd="${AM_ROOT}/kaldi_utils/parallel/spark_ssh.py --queue root.zw01_training.hadoop-speech.cpu --job-name $USER-xvector-train --jumper xr-ai-speech-ttsoffline02"
# export spark_cmd="${AM_ROOT}/kaldi_utils/parallel/spark.py --queue root.zw01_training.hadoop-hdp.cpu_hulk --job-name $USER-sex-recognition"
export train_cmd=$spark_cmd
