#!/usr/bin/env python
import itertools as itt
import os
import subprocess
import tomllib
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Iterable

from pydantic import BaseModel, Field


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", default="run.toml")

    return parser.parse_args()


class RunConfig(BaseModel):
    config: str
    arguments: list[str] = Field(default_factory=list)


class Config(BaseModel):
    n_runs: int
    env_config: str = "sc2"
    runs: list[RunConfig]
    arguments: list[str] = Field(default_factory=list)

    logfile: str = "run.log"


def load_config(filename: str) -> Config:
    with open(filename, "rb") as f:
        config = tomllib.load(f)
    return Config(**config)


def log(filename: str, text: str):
    with open(filename, "a") as f:
        timestamp = datetime.now().strftime("[%F %T]")
        print(f"{timestamp} {text}", file=f)


def make_command(env_config: str, config: str, arguments: list[str]) -> str:
    with_arguments = "with {}".format(" ".join(arguments)) if arguments else ""
    return f"python src/main.py --env-config={env_config} --config={config} {with_arguments}"


def run_command(command: str):
    subprocess.run(command.split())


def make_commands(config: Config) -> Iterable[str]:
    n_runs = range(config.n_runs)

    for _, run in itt.product(n_runs, config.runs):
        arguments = config.arguments + run.arguments
        command = make_command(config.env_config, run.config, arguments)
        command = f"{command}"
        yield command


def main(config: Config):
    if "SC2PATH" not in os.environ:
        home = os.environ["HOME"]
        os.environ["SC2PATH"] = f"{home}/programs/StarCraftII"

    for command in make_commands(config):
        log(config.logfile, command)
        run_command(command)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main(config)