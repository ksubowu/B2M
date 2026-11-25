#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd2/bo/work/cvs/portable_mp123_human_band_transfer_20261110/project
trap 'training_exit=$?; printf "%s\n" "$training_exit" > runs/line_neighborhood_formal_v2_20261119.exit_code' EXIT
export CUDA_VISIBLE_DEVICES=7
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
taskset -c 0,2,4,5,6,9,10,19,22,24,28,29 nice -n 5 \
  /home/bo/bo/miniconda3/envs/bb2m/bin/python -m band_training.finetune_line_neighborhood \
  --init runs/scored_bond_v3_formal_20261117/best_joint.pt \
  --output runs/line_neighborhood_formal_v2_20261119 \
  --epochs 6 --batch-size 8 --workers 4 --lr 0.00001 --patience 2 \
  > runs/line_neighborhood_formal_v2_20261119.log 2>&1
