### Use this to visually inspect whether the clipped raster data files have been created successfully
### Pass file paths as command line arguments
### script loads files and plots them

### example usage: python check_clip.py srtm_raw/srtm_clipped.tif nlcd_raw/nlcd_clipped.tif

import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr

sns.set(font_scale=1.5)

fnames = []
files = []
if len(sys.argv) == 1:
    print("must provide raster files to check as command line arguments")
    sys.exit("exiting... no raster files provided")
else:
    for i in range(1, len(sys.argv)):
        fnames.append(sys.argv[i])
for fname in fnames:
    data_name = fname.split('/')[-1]
    rxr_data = rxr.open_rasterio(fname, masked=True).squeeze()
    f, ax = plt.subplots(figsize=(10, 10), num=1, clear=True)
    rxr_data.plot.imshow()
    ax.set(title="post-clipped " + data_name)
    ax.set_axis_off()
    plt.show()
    plt.cla()

    del rxr_data