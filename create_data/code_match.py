#per earthlab tutorial
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/
#crop-raster-data-with-shapefile-in-python/

### WHAT DOES THIS CODE DO?
### - opens a series of raster datasets
### - clips them to a specified area of interest
### - plots the resulting raster (saves to file)
### - saves the clipped raster data

### REQUIREMENTS
### - Packages
###   - numpy
###   - matplotlib
###   - seaborn
###   - shapely
###   - rioxarray
###   - xarray
###   - geopandas
### - Directory Structure
###   - must provide directory "figures/data" in parent directory of this file's parent directory

### command line arguments:
### raster data path(s)     [comma separated, required]
### output data path(s)     [comma separated, required (same number input paths]
### bounding shape path     [required]
### visualize data          [-v "all", "none", or comma separated indices of data]
### verbosity               [-q {0, 1, 2}, optional (default 2, verbose)]          

# test conda activate code_match; cd work/earthlab/munge_data/; python code_match.py nlcd_raw/reduced_nlcd.tif,srtm_raw/combined.tif nlcd_raw/nlcd_clipped.tif,srtm_raw/srtm_clipped.tif Neon_3D_AOI/NEON_3D_Boundary.shp -v none -q 2

### Usage Example:
### python code_match.py ../raw_data/nlcd_raw/reduced_nlcd.tif,../raw_data/srtm_raw/combined.tif ../raw_data/nlcd_raw/nlcd_clipped.tif,../raw_data/srtm_raw/srtm_clipped.tif ../raw_data/neon_aoi/NEON_3D_Boundary.shp -v none -q 2
### above   reduced_nlcd.tif and combined.tif are nlcd and srtm raster inputs
###         nlcd_clipped.tif and srtm_clipped.tif are the output files
###         Neon_3D_boundary.shp is the bound/area of interest
###         -v all specifies that all graphs of all files should be produced
###         -q 2 specifies maximum verbosity

### default values for verbosity, data visualization
verbosity = 2
visualize = []
vdefault = True

### parse command line arguments
import sys
missing_args = False
init_ok = True
if len(sys.argv) < 4:
    missing_args = True
else:
    #convert inputs to paths
    raster_locs = sys.argv[1].split(',')
    output_locs = sys.argv[2].split(',')
    if len(raster_locs) != len(output_locs):
        print("number of output locations must be the same as number of input locations")
        init_ok = False
    bounding_loc = sys.argv[3]
    if len(sys.argv)%2 == 1:
        missing_args = True
    elif len(sys.argv) <= 8:
        for si in range(4, len(sys.argv), 2):
            if sys.argv[si] == "-v":
                if sys.argv[si+1] == "all":
                    visualize = [ii for ii in range(len(raster_locs))]
                elif sys.argv[si+1] == "none":
                    visualize = []
                else:
                    visualize = [int(idx) for idx in sys.argv[5].split(',')]
                vdefault = False
            elif sys.argv[si] == "-q":
                verbosity = int(sys.argv[si+1])
            else:
                missing_args = True
    else:
        missing_args = True

if vdefault:
    visualize = [ii for ii in range(len(raster_locs))]
if missing_args:
    print("missing command line arguments!")
    print("required: raster_path(s) [comma separated], output_path(s) [comma separated], bounding_path")
    print('optional: -v [visualize data, "all" or comma separated indices], -q [verbosity, 0, 1, 2]')
    init_ok = False
if not init_ok:
    sys.exit("missing or incorrectly used command line arguments")

### done parsing command line arguments
### import packages

if verbosity == 2:
    print ("initializing (importing packages)")
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
#import earthpy as et
#import earthpy.plot as ep
#from osgeo import gdal

if verbosity == 2:
    print ("done importing packages")

sns.set(font_scale=1.5)
#test data types?
#et.data.get_data("colorado-flood")
#os.chdir(os.path.join(et.io.HOME,
#                      'earth-analytics',
#                      'data'))
#old version of cmd inputs
#convert = False
#testplt = [4]
#runon = ["srtm"]

#TODO - send this to its own file
if False:
    print(" - converting srtm .hgt files to .tif")
    prefix1 = "srtm_raw/"
    mergeall = ""
    for n in range(38, 42):
        for w in range(105, 108):
            tname = "N" + str(n) + "W" + str(w)
            os.system("gdal_translate -of GTiff " + prefix1 + tname + ".SRTMGL1.hgt/" + tname + ".hgt " + prefix1 + tname + ".tif")
            mergeall += prefix1 + tname + ".tif "
    print(" - merging .tif files")
    os.system("gdal_merge.py -o srtm_raw/combined.tif " + mergeall)
    print(" - done")



### open datasets provided in command line arguments
### open with rioxarray. Keep masked=True, squeezed
### only open one at a time due to memory challenges
### open aoi first (since we need that for all datas)
if verbosity == 2:
    print("loading AOI")
aoi = gpd.read_file(bounding_loc)
aoi_crs = aoi.crs
for i in range(len(raster_locs)):
    # load provided file into rioxarray
    if verbosity == 1 or verbosity == 2:
        print("Loading Dataset:", raster_locs[i])
    rxr_data = rxr.open_rasterio(raster_locs[i], masked=True).squeeze()
    data_name = raster_locs[i].split('/')[-1]

    # plot unalterered data if called for
    if i in visualize:
        #fig1 = plt.figure(num=1, clear=True)
        if verbosity == 2:
            print("plotting pre-clipped", data_name)
        f, ax = plt.subplots(figsize=(10,10), num=1, clear=True)
        rxr_data.plot.imshow()
        ax.set(title="pre-clipped" + data_name)
        ax.set_axis_off()
        plt.show()
        plt.cla()

    # get crs of data
    data_crs = rxr_data.rio.crs

    # plot area of interest superimposed on image if called for
    if i in visualize:
        f, ax = plt.subplots(figsize=(9, 12), num=1, clear=True)
        rxr_data.plot.imshow(ax=ax)
        aoi.plot(ax=ax, alpha=0.8)
        ax.set(title="Area of Interest overlaid on " + data_name)
        ax.set_axis_off()
        plt.show()
        plt.cla()

    # clip data to area of interest
    if verbosity == 2:
        print("clipping", data_name, "to area of interest")
    data_clipped = rxr_data.rio.clip(aoi.geometry.apply(mapping))

    # clear unclipped data from memory (avoid crashing)
    if verbosity == 2:
        print("removing unclipped data from memory")
    del rxr_data

    # save clipped data to geotif file
    if ".tif" not in output_locs[i]:
        output_locs[i] += ".tif"
    data_clipped.rio.to_raster(output_locs[i])
    if verbosity == 2:
        print("saved data to geotif")

    # plot raster data clipped to area of interest
    #plt.figure()
    f, ax = plt.subplots(figsize=(9, 12), num=1, clear=True)
    data_clipped.plot(ax=ax)
    ax.set(title= data_name + " Data Clipped to Area of Interest")
    ax.set_axis_off()
    fdataname = data_name.split(".")[0]
    plt.savefig("../figures/data/" + fdataname + ".png")
    if i in visualize:
        plt.show()
    plt.close()
    plt.cla()

    #clean memory
    if verbosity == 2:
        print("removing files from memory")
    del data_clipped

    if verbosity == 1 or verbosity == 2:
        print("Completed Dataset:", raster_locs[i])
if verbosity == 1 or verbosity == 2:
    print("Done, exiting.")

"""
print("opening nlcd and srtm data...")
if "nlcd" in runon:
    nlcd = rxr.open_rasterio("nlcd_raw/clipped_nlcd.tif", masked=True).squeeze()
if "srtm" in runon:
    srtm = rxr.open_rasterio("srtm_raw/combined.tif", masked=True).squeeze()
"""
"""
if 1 in testplt:
    if "srtm" in runon:
        print("test: plotting srtm data")
        f, ax = plt.subplots(figsize=(10,5))
        srtm.plot.imshow()
        ax.set(title="srtm data")
        ax.set_axis_off()
        plt.show()
"""

"""print("loading gedi shapefile...")
#aoi = "GEDI_2B_clean/GEDI_2B_clean.shp"
aoi = "Neon_3D_AOI/NEON_3D_Boundary.shp"
ult_aoi = gpd.read_file(aoi)
aoi_crs = ult_aoi.crs
if "nlcd" in runon:
    nlcd_crs = nlcd.rio.crs
if "srtm" in runon:
    srtm_crs = srtm.rio.crs
if "srtm" in runon and "nlcd" in runon:
    print(aoi_crs, nlcd_crs, srtm_crs)
if 2 in testplt:
    if "srtm" in runon:
        fig, ax = plt.subplots(figsize=(6,6))
        ult_aoi.plot(ax=ax)
        ax.set_title("gedi aoi baby")
        plt.show()

if 3 in testplt:
    if "srtm" in runon:
        f, ax = plt.subplots(figsize=(9,12))
        srtm.plot.imshow(ax=ax)
        ult_aoi.plot(ax=ax, alpha=0.8)
        ax.set(title="aoi overlay on srtm data")
        ax.set_axis_off()
        plt.show()

print("clipping data to shape")
if "nlcd" in runon:
    nlcd_clipped = nlcd.rio.clip(ult_aoi.geometry.apply(mapping))
    #nlcd_gdal = gdal.Open("nlcd_raw/nlcd_full.tif")
    #nlcd_arr = nlcd_gdal.GetRasterBand(1).ReadAsArray()
    #plt.imshow(array)
    #plt.colorbar()

if "srtm" in runon:
    srtm_clipped = srtm.rio.clip(ult_aoi.geometry.apply(mapping))

print("plotting...")
if 4 in testplt:
    if "srtm" in runon:
        plt.figure()
        f, ax = plt.subplots(figsize=(9,12))
        srtm_clipped.plot(ax=ax)
        ax.set(title="SRTM data clipped to area of interest")
        ax.set_axis_off()
        plt.savefig("figures/data/srtm_clipped.png")
        #plt.show()
        plt.close()
        plt.cla()
    if "nlcd" in runon:
        plt.figure()
        f, ax = plt.subplots(figsize=(9, 12))
        nlcd_clipped.plot(ax=ax)
        #ult_aoi.plot(ax=ax, alpha=0.8)
        ax.set(title="NLCD data clipped to area of interest")
        ax.set_axis_off()
        plt.savefig("figures/data/nlcd_clipped.png")
        #plt.show()
        plt.close()
"""
#save files...
#figure out code to clip srtm to nlcd...?
