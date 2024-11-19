#!/bin/bash

seeds=(0 1 2)

configs=("qfix_sum" "qfix_sum_alt")

for config in "${configs[@]}"; do
  for seed in "${seeds[@]}"; do
    echo "Running with config=${config}, seed=${seed}"
    python3 src/main.py --config=${config} --env-config=sc2 with env_args.map_name=5m_vs_6m seed=${seed}
  done
done