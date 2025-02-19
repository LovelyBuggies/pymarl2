#!/bin/bash

# Install tqdm if not already installed
pip install tqdm > /dev/null 2>&1

# Define the lists of values
algs=("vdn" "qmix" "qplex")
problems=("one_step_matrix_game")

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
        for i in {1..5}; do
                # Get current timestamp as seed
                seed=$(date +%s)

                # Construct name and group
                name="${alg}"
                group="$problem"

                # Construct full command
                CMD="$BASE_CMD --env-config=one_step_matrix_game --config=$alg with t_max=20000 use_wandb=True name=$name group=$group env_args.map_name=$problem"

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

# Close tqdm progress bar at the end
python -c "from tqdm import tqdm; pbar = tqdm(total=$TOTAL_TASKS, dynamic_ncols=True); pbar.update($TOTAL_TASKS); pbar.close()"

echo "All experiments completed!"