#!/bin/bash

source ./source-this.sh

maps_path="$SC2PATH/Maps"
mkdir -p "$maps_path"

echo "installing SMAC maps"
(
  cd "$maps_path" || exit
  wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
  unzip -o SMAC_Maps.zip 'SMAC_Maps/*.SC2Map'
  rm SMAC_Maps.zip
)

