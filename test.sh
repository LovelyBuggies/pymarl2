#!/bin/bash

source ./source-this.sh

nruns=3
configs=(
  vdn
  qmix
  qfix_sum
  qfix_mono
  qfix_sum_alt
)

env_config=sc2

logfile=./test-status.log
timestamp=$(date +'%F %T')
echo "$timestamp NEW TEST" >> $logfile

for _ in $(seq $nruns); do
for config in "${configs[@]}"; do

  arguments=(
    env_args.map_name=5m_vs_6m
    "name=$config"
  )

  logging_arguments=(
    # entity=abaisero
    # use_wandb=False
  )

  python_arguments=()
  # python_arguments=(-m ipdb -c continue)

  timestamp=$(date +'%F %T')
  command="python ${python_arguments[*]} src/main.py --config=$config --env-config=$env_config with ${arguments[*]} ${logging_arguments[*]}"

  echo "$timestamp $command" >> $logfile
  $command

  # kill-starcraft.sh

done
done
