import json
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.signal import savgol_filter
import argparse
import os

# Set up argument parsing
parser = argparse.ArgumentParser(description="Plot battle won mean for various algorithms across multiple maps.")
parser.add_argument('--maps', type=str, nargs='+', required=True, help='List of map names')
parser.add_argument('--smooth', action='store_true', help='Apply smoothing filter')

args = parser.parse_args()
map_names = args.maps
smooth = args.smooth

def plot_algorithm(file_path, label, color, alpha=1):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        if not smooth:
            battle_won_mean = data['battle_won_mean']
        else:
            battle_won_mean = savgol_filter(data['battle_won_mean'], window_length=15, polyorder=2)
            battle_won_mean[battle_won_mean < 0] = 0
        steps = [i / 100 for i in range(len(battle_won_mean))]
        plt.plot(
            steps,
            battle_won_mean,
            lw=2,
            label=label,
            c=color,
            alpha=alpha,
        )
        print(f"{label} Battle Won Mean: {max(battle_won_mean[-120:]) * 100:.2f}%")
    else:
        print(f"File not found: {file_path}")

# Calculate rows and columns for subplots
num_maps = len(map_names)
columns = 3
rows = (num_maps + columns - 1) // columns  # Ceiling division to determine number of rows

fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(24, 8 * rows), constrained_layout=True)

# Flatten axes array if needed
if isinstance(axes, plt.Axes):
    axes = [axes]
else:
    axes = axes.flatten()

for i, (ax, map_name) in enumerate(zip(axes, map_names)):
    plt.sca(ax)  # Set the current axis to the subplot for the current map
    ax.set_title(f"{map_name}", fontsize=24)

    # Plotting each algorithm for the current map
    plot_algorithm(f'./results/sacred/{map_name}/qmix/best/info.json', "Qmix", "purple")
    plot_algorithm(f'./results/sacred/{map_name}/qplex/best/info.json', "Qplex", "goldenrod")
    plot_algorithm(f'./results/sacred/{map_name}/qfix_sum/best/info.json', r"$Qfix-sum$", "indianred")
    plot_algorithm(f'./results/sacred/{map_name}/qfix_mono/best/info.json', r"$Qfix-mono$", "steelblue")
    plot_algorithm(f'./results/sacred/{map_name}/qfix_sum_alt/best/info.json', "Qfix-sum-alt", "darkgreen")

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel('Total Sampling Steps (mil)', fontsize=18, labelpad=6)
    ax.set_ylabel('Battle Won Mean', fontsize=18, labelpad=6)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(loc=2, labelspacing=1., handleheight=2, prop={"size": 12})
    ax.grid()

# Hide any unused subplots if the number of maps is not a multiple of columns
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
