#!/usr/bin/env bash
set -uo pipefail
cd /mnt/hdd2/bo/work/cvs/portable_mp123_human_band_transfer_20261110/project
exec >> runs/scored_bond_v3_formal_20261117.log 2>&1
date -Iseconds
export BB2M_GPU=0
export BB2M_OUTPUT=runs/scored_bond_v3_formal_20261117
export NUMEXPR_NUM_THREADS=2
taskset -c 22,35,36,42,48,50,61,62,72,77,89,96 nice -n 5 bash scripts/train_scored_bonds.sh
training_exit=$?
printf '%s\n' "$training_exit" > runs/scored_bond_v3_formal_20261117.exit_code
date -Iseconds
exit "$training_exit"
