#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd2/bo/work/cvs/portable_mp123_human_band_transfer_20261110/project
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
trap 'code=$?; echo "$code" > /mnt/hdd2/bo/work/cvs/portable_mp123_human_band_transfer_20261110/project/runs/v6_recovery_20261124/supervisor_exit_code' EXIT
/home/bo/bo/miniconda3/envs/bb2m/bin/python -u -m band_training.run_v6_recovery --root /mnt/hdd2/bo/work/cvs/portable_mp123_human_band_transfer_20261110/project/runs/v6_recovery_20261124 > /mnt/hdd2/bo/work/cvs/portable_mp123_human_band_transfer_20261110/project/runs/v6_recovery_20261124/supervisor.log 2>&1
