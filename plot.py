#!/usr/bin/env python
import argparse
import glob
import itertools as itt
import json
import os
from operator import itemgetter
from typing import NamedTuple

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so

from config import Config, KeysConfig, load_config

sns.set_theme()


class LabeledRun(NamedTuple):
    label: str
    run: pd.DataFrame


def load_run(filename: str, config: KeysConfig) -> pd.DataFrame:
    print(f"loading {filename=}")
    with open(filename) as f:
        try:
            data = json.load(f)
        except Exception as error:
            print(error)
            raise error

    keymap = {
        config.x: f"{config.y}_T",
        config.y: config.y,
    }
    data = {newkey: data[oldkey] for newkey, oldkey in keymap.items()}

    if "return_mean" in data:
        data["return_mean"] = [d["value"] for d in data["return_mean"]]

    return pd.DataFrame(data)


def load_runs(config: Config) -> list[LabeledRun]:
    runs: list[LabeledRun] = []
    for dconfig in config.data:
        datapath = os.path.expanduser(dconfig.path)
        datapath = datapath.format(map=config.map, label=dconfig.label)
        print(f"globbing {datapath=}")

        label = dconfig.label if dconfig.relabel is None else dconfig.relabel

        for filename in sorted(glob.glob(datapath)):
            run = load_run(filename, config.keys)
            run = LabeledRun(label, run)
            runs.append(run)

    return runs


def interpolate_run(
    run: pd.DataFrame,
    timesteps: list[int],
    keys: KeysConfig,
) -> pd.DataFrame:
    run = run.set_index(keys.x)
    index = run.index.values.tolist()
    index = sorted(set(index + timesteps))
    run = run.reindex(index)
    run = run.interpolate("linear", limit_direction="both")
    run = run.loc[timesteps]
    run = run.reset_index()

    return run


def interpolate_runs(
    labeled_runs: list[LabeledRun],
    keys: KeysConfig,
    num: int,
) -> list[LabeledRun]:
    xmin = 0
    xmax = max(run[keys.x].max().item() for _, run in labeled_runs)

    timesteps = np.linspace(xmin, xmax, num)
    timesteps = timesteps.astype(int).tolist()

    interpolated = []
    for label, run in labeled_runs:
        run = interpolate_run(run, timesteps, keys)
        interpolated.append(LabeledRun(label, run))

    return interpolated


def combine_runs(runs: list[LabeledRun], keys: KeysConfig) -> pd.DataFrame:
    combined = []

    runs = sorted(runs, key=itemgetter(0))
    for _, group in itt.groupby(runs, key=itemgetter(0)):
        for i, (group_label, run) in enumerate(group):
            unit_label = f"{group_label}-{i}"

            run[keys.group] = group_label
            run[keys.unit] = unit_label
            combined.append(run)

    return pd.concat(combined)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.toml")
    return parser.parse_args()


def load_data(config) -> pd.DataFrame:
    print(f"loading runs {config.map=}")
    runs = load_runs(config)

    print("preparing data")
    runs = interpolate_runs(runs, config.keys, 100)
    data = combine_runs(runs, config.keys)

    return data


def plot(
    data: pd.DataFrame,
    keys: KeysConfig,
    title: str,
    *,
    counts: bool = False,
    move_legend: bool = False,
):
    if counts:
        labels = data[keys.group].unique().tolist()
        counts_map = data.groupby(keys.group)[keys.unit].nunique().to_dict()
        data[keys.group] = data[keys.group].replace(
            {label: f"{label} ({counts_map[label]})" for label in labels}
        )

    fig, ax = plt.subplots()
    plotter = (
        so.Plot(data, x=keys.x, y=keys.y, color=keys.group)
        .add(so.Line(), so.Agg("mean"))
        .add(so.Band(edgealpha=0.25, edgewidth=1), so.Est(errorbar="se"))
        # .add(so.Band(edgealpha=0.25, edgewidth=1), so.Est(errorbar=("pi", 100)))
        .scale(
            x=so.Continuous().label(matplotlib.ticker.EngFormatter()),
            # y=so.Continuous().label(matplotlib.ticker.PercentFormatter(1.0, is_latex=True)),
            y=so.Continuous().label(like="{x:1.0%}"),
        )
        .limit(y=(0, 1))
        .label(title=title)
        .on(ax)
        .plot()
    )

    if move_legend:
        legend = fig.legends.pop(0)
        handles: list = legend.legend_handles
        labels = [t.get_text() for t in legend.texts]
        ax.legend(handles, labels)
        sns.move_legend(ax, loc="best")

    plotter.save(f"{title}.pdf", bbox_inches="tight")


def main():
    args = parse_args()
    config = load_config(args.config)

    data = load_data(config)

    print(f"plotting {config.map=}")
    plot(
        data,
        config.keys,
        title=config.map,
        counts=True,
    )


if __name__ == "__main__":
    main()
