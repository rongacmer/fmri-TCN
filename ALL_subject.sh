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
for i in `cat $1`;
do
  python -u niitobrainnpy.py \
  --nii $2/${i}.nii \
  --name ${i} \
  --output_filename $3 \
  2>&1 | tee -a $LOG
done