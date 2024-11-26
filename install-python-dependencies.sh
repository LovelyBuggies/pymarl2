#!/bin/bash

install_packages=(
  git+https://github.com/oxwhirl/smac.git@26f4c4e4d1ebeaf42ecc2d0af32fac0774ccc678
  gym==0.11
  imageio
  matplotlib
  numpy
  probscale
  protobuf==3.19.5
  pydantic
  pygame
  pytest
  pyyaml
  sacred
  scipy
  seaborn
  snakeviz
  tensorboard-logger
  torch
)

dev_packages=(
  ipdb
  ipython
)

python -m pip install "${install_packages[@]}"
python -m pip install "${dev_packages[@]}"
