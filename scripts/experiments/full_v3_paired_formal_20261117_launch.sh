#!/usr/bin/env bash
set -uo pipefail
cd /mnt/hdd2/bo/work/cvs/portable_mp123_human_band_transfer_20261110/project
exec >> runs/full_v3_paired_formal_20261117.log 2>&1
date -Iseconds
export BB2M_GPU=0
export BB2M_OUTPUT=runs/full_v3_paired_formal_20261117
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
taskset -c 0,2,4,5,6,9,10,19,22,24,28,29 nice -n 5 bash scripts/train_full_v3.sh
training_exit=$?
printf '%s\n' "$training_exit" > runs/full_v3_paired_formal_20261117.exit_code
date -Iseconds
exit "$training_exit"
