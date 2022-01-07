#!/bin/bash

set -e
set -x

CUDA=$1 #使用的gpu
threshold=$2

#set CUDA_VISIBLE_DEVICES=$CUDA
threshold_array=(${threshold//,/ })

for i in ${threshold_array[@]}
do
  echo "/home/gpusharedata/rhb/fmri/output/P_network_threshold_$i"
  echo /home/gpusharedata/rhb/fmri/model/AD_NC_COF_GCN_MLP_threshold_${i}
  echo $i
  CUDA_VISIBLE_DEVICES="$CUDA" python -u  main.py --data_root /home/gpusharedata/rhb/fmri/output/P_network_threshold_$i --threshold $i  --model_dir /home/gpusharedata/rhb/fmri/model/AD_NC_COF_GCN_MLP_threshold_${i}
done
