#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail
export PYTHONUNBUFFERED="True"
if [ ! -d ../logs ];then
  mkdir ../logs
fi
LOG=" ../logs/$(date +"%Y-%m-%d_%H-%M-%S").txt"
for i in {0..10};
do
  python -u mainVAE.py \
  --data_root /home/sharedata/gpuhome/rhb/fmri/no_mean_data/PostProcessing/AAL/ROI_${i} \
  --model_dir /home/gpusharedata/rhb/fmri/AAL_model/ \
  --batch_size 200 \
  --epochs 1000 \
  --ROI_num ${i} \
  --trainortest train \
  2>&1 | tee -a $LOG
done
