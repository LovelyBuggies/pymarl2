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
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--debug", action="store_true")

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


def make_commands(config: Config) -> Iterable[str]:
    run_ids = range(config.n_runs)

    for _, run in itt.product(run_ids, config.runs):
        arguments = config.arguments + run.arguments
        yield make_command(config.env_config, run.config, arguments)


def run_command(command: str, **kwargs):
    subprocess.run(command.split(), **kwargs)


def make_env(args: Namespace) -> dict:
    env = {}

    if "SC2PATH" not in env:
        home = os.environ["HOME"]
        env["SC2PATH"] = f"{home}/programs/StarCraftII"

    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.debug:
        print("Settig enviroment:")
        for k, v in env.items():
            print(f"- {k} = {v}")
        print()

    env = dict(os.environ, **env)

    return env


def main(args: Namespace):
    env = make_env(args)
    config = load_config(args.config)

    for command in make_commands(config):
        if args.debug:
            command = f"echo {command}"

        log(config.logfile, command)
        run_command(command, env=env)


if __name__ == "__main__":
    args = parse_args()
    main(args)
