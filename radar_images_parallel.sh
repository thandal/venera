#!/bin/bash

ROOT_PREFIX="data_venus/arecibo_radar/pds-geosciences.wustl.edu/venus/arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/"
DATA_PREFIX="$ROOT_PREFIX/data/"

## 1. Convert to notebook to .py
jupyter nbconvert --to python radar_images.ipynb

## 2. Run the .py in parallel:
## TODO: reduce memory usage to allow more parallel instantiations! (Can only run 4 with ~32 GB of RAM)
#! ls -1 $DATA_PREFIX/*.img | xargs -n 1 basename | xargs -n 1 -P 4 python3 $script
ls -1 $DATA_PREFIX/*1988*.img | xargs -n 1 basename | xargs -n 1 -P 4 python3 radar_images.py
#! ls -1 *.img | xargs -n 1 -P 4 python3 ${cwd}/radar_images.py

