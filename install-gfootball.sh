#!/bin/bash

install_apt_packages=(
  build-essential
  cmake
  git
  libboost-all-dev
  libdirectfb-dev
  libgl1-mesa-dev
  libsdl2-dev
  libsdl2-gfx-dev
  libsdl2-image-dev
  libsdl2-ttf-dev
  libst-dev
  mesa-utils
  x11vnc
  xvfb
)

sudo apt-get install "${install_apt_packages[@]}"

upgrade_python_packages=(
  psutil
  pytest
  wheel
)

install_python_packages=(
  gfootball==2.10.2
  gym==0.11
)

python -m pip install --upgrade "${upgrade_python_packages[@]}"
python -m pip install "${install_python_packages[@]}"
