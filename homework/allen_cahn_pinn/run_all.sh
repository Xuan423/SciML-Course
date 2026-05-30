#!/usr/bin/env bash
set -euo pipefail

source ~/miniforge/etc/profile.d/conda.sh
conda activate phmbench

python train.py --run_all
