#!/bin/sh
cd data_venus/arecibo_radar/pds-geosciences.wustl.edu/venus/arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/GLOBAL_TRIAGE/
ffmpeg \
 -f image2 \
 -r 10 \
 -pattern_type glob -i 'venus*.png' \
 -c:v libx264 \
 -s 2048x1024 \
 -vf tmix=frames=8:weights="8 8 4 4 2 2 1 1" \
 video_10fps_tmix_8.mp4