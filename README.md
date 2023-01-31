# Venera : Venus Planetary Radar Astronomy
## Doppler-Delay Image Processing

[Planetary Radar Astronomy](https://en.wikipedia.org/wiki/Radar_astronomy) is the use of active radar to map celestial bodies (the Moon, planets, asteroids). From 1988 to 2020, Arecibo intermittently mapped Venus, with [data archived here](https://pds-geosciences.wustl.edu/missions/venus_radar/index.htm) (citation: Bruce A. Campbell and Donald B. Campbell 2022 Planet. Sci. J. 3 55, doi:10.3847/PSJ/ac4f43.).

The code in this repository is an exploration in processing this data, projecting it into Venus global coordinates, and combining multiple observations into videos or super-resolution images.

WORK IN PROGRESS -- the code is still under development, and has been hacked up to produce various outputs.

- Use `wget_venus_data.sh` to download data, for example `wget_venus_data.sh ocp_2015` to get all the OCP imgs from 2015. 
- Use process_radar_images.ipynb to do processing (or use parallel_process_radar_images.sh to run it in parallel -- but you need a lot of memory!)
- Use process_video.ipynb to assemble videos.

## Preliminary Data Products

- [High res projected videos (Google Drive)](https://drive.google.com/drive/folders/11YsTmb8AydKsmTp8NOlG0jSVC8TS2cPJ)
  - Example: [Cumulative average, 1988-2020 2000x2000 (5 MB)](https://drive.google.com/file/d/11d1ctpYEdp0TgNgoxYlh8hmKDqZ-8FUq)
- [Super-resolution projected photos (Google Drive)](https://drive.google.com/drive/folders/11WIMnZPHnMQXcip6fFitsQyXO6jqbWdo)
  - Example: [All 2015 observations combined (118 MB)](https://drive.google.com/file/d/11qP2Xkku4XNgmMXH052p0ebfdr-SKs_9)
  
Note: due to observational constraints, all observations are basically from one hemisphere of Venus (the side closest to the Earth at nearest approach).

Example doppler-delay image:
![venus_ocp_20150813_161747_small.png](/figures/venus_ocp_20150813_161747_small.png)
