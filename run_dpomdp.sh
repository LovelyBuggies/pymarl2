#!/bin/bash

# Install tqdm if not already installed
pip install tqdm > /dev/null 2>&1

# Define the lists of values
algs=("vdn" "qmix" "qplex")
problems=("grid_small" "dtiger" "recycling" "firefighting")
horizons=(2 3 4)

# Calculate total number of tasks
TOTAL_TASKS=$(( ${#algs[@]} * ${#problems[@]} * ${#horizons[@]} ))

# Base command
BASE_CMD="python src/main.py"

# Initialize tqdm progress bar
echo "Running experiments..."
python -c "from tqdm import tqdm; global pbar; pbar = tqdm(total=$TOTAL_TASKS, dynamic_ncols=True)" &

TASK_COUNT=0
for alg in "${algs[@]}"; do
    for problem in "${problems[@]}"; do
        for horizon in "${horizons[@]}"; do
            for i in {1..5}; do
                # Get current timestamp as seed
                seed=$(date +%s)

                # Construct name and group
                name="${alg}_horizon=${horizon}_seed=${seed}"
                group="$problem"

                # Construct full command
                CMD="$BASE_CMD --env-config=dpomdp --config=$alg with t_max=500000 use_wandb=True name=$name group=$group env_args.episode_limit=$horizon env_args.map_name=$problem env_args.seed=$seed"

                # Print and execute command
                echo "Running: $CMD"
                eval $CMD

                # If the command fails, continue to the next iteration
                if [ $? -ne 0 ]; then
                    echo "Command failed: $CMD. Moving to next."
                    continue
                fi

                # Update tqdm progress bar
                TASK_COUNT=$((TASK_COUNT + 1))
                python -c "from tqdm import tqdm; tqdm.write(''); pbar = tqdm(total=$TOTAL_TASKS, dynamic_ncols=True); pbar.update($TASK_COUNT); pbar.close()"
            done
        done
    done
done

# Close tqdm progress bar at the end
python -c "from tqdm import tqdm; pbar = tqdm(total=$TOTAL_TASKS, dynamic_ncols=True); pbar.update($TOTAL_TASKS); pbar.close()"

echo "All experiments completed!"