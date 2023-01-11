#!/bin/sh

# Usage:
#  ./wget_data_venus.sh         <-- get *all* the data
#  ./wget_data_venus.sh 2015    <-- get all the 2015 data

mkdir -p data_venus/arecibo_radar
cd data_venus/arecibo_radar

wget -r -c -np -nc --accept "*$1*" https://pds-geosciences.wustl.edu/venus/arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/
