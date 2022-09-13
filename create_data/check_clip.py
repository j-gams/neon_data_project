### Written by Jerry Gammie @j-gams

### WHAT DOES THIS CODE DO?
### - opens specified geotif files
### - plots them
### - Use this to visually inspect whether the clipped raster data files have been created successfully

### REQUIREMENTS
### - Packages
###   - matplotlib
###   - seaborn
###   - shapely
###   - rioxarray
###   - xarray

### command line arguments:
### raster data path(s)     [list as many file paths as needed]

### Usage Example:
### python check_clip.py ../raw_data/srtm_raw/srtm_clipped.tif ../raw_data/nlcd_raw/nlcd_clipped.tif

import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
#from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr

sns.set(font_scale=1.5)

### get file names from command line arguments
fnames = []
files = []
if len(sys.argv) == 1:
    print("must provide raster files to check as command line arguments")
    sys.exit("exiting... no raster files provided")
else:
    for i in range(1, len(sys.argv)):
        fnames.append(sys.argv[i])

### load each file sequentially. Plot it. Remove it from memory
for fname in fnames:
    data_name = fname.split('/')[-1]
    rxr_data = rxr.open_rasterio(fname, masked=True).squeeze()
    f, ax = plt.subplots(figsize=(8, 16), num=1, clear=True)
    rxr_data.plot.imshow()
    ax.set(title="post-clipped " + data_name)
    ax.set_axis_off()
    plt.show()
    plt.cla()

    del rxr_data
    del f
    del ax