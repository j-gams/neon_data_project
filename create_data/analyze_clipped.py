### WHAT DOES THIS CODE DO?
### - opens a series of raster data files
### - opens a shapefile containing points
### - computes information about the shapefiles at the points
### - makes some graphs

### REQUIREMENTS
### - Packages
###   - numpy
###   - matplotlib
###   - seaborn
###   - shapely
###   - rioxarray
###   - xarray
###   - geopandas
###   - osgeo
###   - shapefile
###   - pyproj

### command line arguments
### raster data path(s)             [comma separated, required]
### points of interest shapefile    [file path, required]
### run_filters                     [-f comma separated indices of datasets to run filters on, optional (default none)]
### verbosity                       [-q {0, 1, 2}, optional (default 2, verbose)]
### test_mode                       [-t int, optional (look at the first n points, -1 for all points. default -1)]

### NOTES
### - this code makes some pretty big assumptions about the structure of raster and point data
### - to avoid issues, everything should be in "EPSG:4326" or the code below should be changed
### - my understanding is that this is the default crs for the SRTM and NLCD data this code is written for
### - gdal represents raster data as a grid starting at an upper left x and y coordinate,
###   a pixel width and height, and two parameters for the angle of the pixels
###   This code assumes that both angle parameters are ZERO
###   If this is not the case, may god have mercy on your soul
### - spatial fileters are intended for topographical data where gradients etc make sense to calculate

### NOTE TO SELF
### could probably combine everything to only have to deal with gdal instead of gdal and rxr

### Usage Example:
### python analyze_clipped.py ../raw_data/srtm_raw/srtm_clipped.tif,../raw_data/nlcd_raw/nlcd_clipped.tif,../raw_data/ecos_wue/WUE_Median_Composite_AOI.tif ../raw_data/gedi_pts/GEDI_2B_clean.shp -f 0 -t 200

### parse command line arguments
import sys
init_ok = True
raster_locs = []
poi_loc = ""
filter_on = []
verbosity = 2
test_mode = -1
if len(sys.argv) < 3:
    init_ok = False
else:
    raster_locs = sys.argv[1].split(",")
    poi_loc = sys.argv[2]
    for i in range(3, len(sys.argv), 2):
        if sys.argv[i] == "-f":
            filter_on = [int(ii) for ii in sys.argv[i+1].split(",")]
        elif sys.argv[i] == "-q":
            verbosity = int(sys.argv[i+1])
        elif sys.argv[i] == "-t":
            test_mode = int(sys.argv[i+1])
if not init_ok:
    print("required: raster path(s) [comma separated], points_of_interest path")
    print("optional: -f [apply filters, indices of datasets], -q [verbosity, 0, 1, 2]")
    sys.exit("missing or incorrect command line arguments!")

### function to handle printing with verbosity settings more easily
def qprint(instr, power):
    if power <= verbosity:
        print(instr)

### done parsing command line arguments
### import packages
qprint("initializing (importing packages)", 2)
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array
from osgeo import gdalconst
import shapefile
from pyproj import Transformer

#conda activate code_match; cd work/earthlab/munge_data/; python analyze_clipped.py
#stolen helper

### in geotif, raster data is stored as upper left pixel coordinates, pixel dimension, and rotation of each basis
### so we can get value at coordinate (x, y) from looking at pixel at
### (x - UL_x)/size_x
### (UL_y - y)/size_y
### here these are switched because in the geographic crs, increasing coordinates go up and to the right
### however within the matrix they go down and to the right, so the sign must be reversed for y
### this function converts from coordinate to array index
def get_index(coordx, coordy, ulht, ulvt, psh, psv):
    idxx = int((coordx - ulht)/psh)
    idxy = int((ulvt - coordy)/psv)
    return idxx, idxy

flatfilter_threshold = 4*8

### these are the filters to be used for aspect analysis
filters = [np.array([[1,  1, 1.0],
                     [0,  0, 0.0],
                     [-1,-1,-1.0]]),
           np.array([[0,   1,  1.4,],
                     [-1,  0,  1],
                     [-1.4,-1, 0]]),
           np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]]),
           np.array([[-1.4, -1, 0],
                     [-1, 0, 1],
                     [0, 1, 1.4]])
           ]

qprint("done importing packages", 2)


qprint("loading points-of-interest data", 2)
### read in the point of interest shape file
poi_data = shapefile.Reader(poi_loc)
### get the data records that contain coordinates, etc.
poi_recs = poi_data.shapeRecords()
qprint("done loading points-of-interest data", 2)

### now sequentially load each raster file and compute distribution of values, etc on it
for i in range(len(raster_locs)):
    rloc = raster_locs[i]

    ### load in raster file with rioxarray for easy computation of value distribution over raster data
    tdataname = rloc.split("/")[-1]
    qprint("loading data " + tdataname + " (rioxarray)", 1)
    rdata = rxr.open_rasterio(rloc, masked=True).squeeze()

    ### if we need to run filters over the data, also load into gdal to make this possible
    if i in filter_on:
        qprint("loading data " + tdataname + " (gdal)", 1)
        gdata = gdal.Open(rloc, gdal.GA_ReadOnly)
        gdata_rsize = (gdata.RasterXSize, gdata.RasterYSize)
        ulh, pxh, _, ulv, _, pxv = gdata.GetGeoTransform()
        pxv = abs(pxv)
        qprint(tdataname + " crs information", 2)
        qprint("crs parameters: " + str(ulh) + " " + str(ulv) + " " + str(pxh) + " " + str(pxv), 2)
        qprint("raster size: " + str(gdata_rsize), 2)
        gdata_npar = gdata.ReadAsArray().transpose()

    ### get the number of points of interest to look at
    ### this is determined by the test_mode command line argument. -1 (default) means run over every point
    if test_mode < 0:
        cut = len(poi_data)
    else:
        cut = test_mode
    qprint("running on first " + str(cut) + " points of interest", 1)

    ### get the transformer for the rioxarry raster data
    rtransformer = Transformer.from_crs("EPSG:4326", rdata.rio.crs, always_xy=True)

    ### list to store basic information at points of interest
    values_of_interest = []
    if i in filter_on:
        ### np array to store information from filters
        np_voi = np.zeros((cut, len(filters)))
        flat_filter = np.zeros(cut)

    ### math for progress bar
    progress_unit = cut // 50
    if verbosity > 0:
        print("progress 0/50 ", end="", flush = True)

    ### get information at every point of interest
    for j in range(cut):
        ### progress bar stuff
        if verbosity > 0 and (j+1) % progress_unit == 0:
            print("-", end="", flush=True)

        ### get coordinates of point of interest from the shapefile record
        a, b = [poi_recs[j].record[2], poi_recs[j].record[3]]

        ### transform the point's coordinates to the appropriate crs for the raster data
        tx, ty = rtransformer.transform(b, a)

        ### get the raster value at the point of interest. simple
        values_of_interest.append(rdata.sel(x=tx, y=ty, method='nearest').values)

        ### if filters are on for this dataset, run every filter over every point
        if i in filter_on:
            ### get indices in array storing raster data from the coordinates
            ix, iy = get_index(tx, ty, ulh, ulv, pxh, pxv)

            ### run over every filter
            for f in range(len(filters)):
                ### filters are 3x3... so we need to look at every point with coordinates within 1 of the poi
                for oi in [-1, 0, 1]:
                    for oj in [-1, 0, 1]:
                        nx = oi + ix
                        ny = oj + iy
                        ### make sure everything is in-bounds
                        if nx >= 0 and ny >= 0 and nx < gdata_rsize[0] and ny < gdata_rsize[1]:
                            #print(j, f, oi, oj)
                            np_voi[j, f] += gdata_npar[nx, ny] * filters[f][oi, oj]

                            ### to get some sense for how flat the terrain is
                            if f == 0:
                                flat_filter[j] += abs(gdata_npar[nx, ny] - gdata_npar[ix, iy])


    ### now we have gathered all the information we wanted
    qprint("---> 50/50 (done)", 1)

    ### make basic graph
    plt.figure()
    plt.hist(values_of_interest)
    plt.title(tdataname + " Values at Points of Interest")
    plt.savefig("../figures/gedi_distributions/" + tdataname.split(".")[0] + ".png")
    plt.show()
    plt.cla()
    plt.close()

    ### find most meaningful filter index of largest value of first 4 filters
    ### combination of the last 2 are designed to find flat areas
    if i in filter_on:
        max_abs = np.argmax(np.abs(np_voi), axis=1)
        #ma_sign = np.zeros(len(max_abs))
        total_dir = np.zeros(cut)
        for i in range(cut):
            if np_voi[i, max_abs[i]] != 0:
                total_dir[i] = max_abs[i] * (np_voi[i,max_abs[i]]/abs(np_voi[i,max_abs[i]])) 
        #ma_sign = np_voi[:, max_abs]/np.abs(np_voi[:, max_abs])
        #print(max_abs.shape)
        #print(ma_sign.shape
        #print(total_dir.shape)
        #total_dir = ma_sign * max_abs

        plt.figure()
        plt.hist(total_dir)
        plt.title(tdataname + " Aspect")
        plt.savefig("../figures/gedi_distributions/aspect_" + tdataname.split(".")[0] + ".png")
        plt.show()
        plt.cla()
        plt.close()

        plt.figure()
        plt.hist(flat_filter)
        plt.title(tdataname + " Flatness")
        plt.savefig("../figures/gedi_distributions/flatness_" + tdataname.split(".")[0] + ".png")
        plt.show()
        plt.cla()
        plt.close()

    ### done with graphs
    qprint("done with " + tdataname, 1)
    ### clean up memory
    del rdata
    del values_of_interest
    if i in filter_on:
        del gdata
        del gdata_npar
        del np_voi
        del flat_filter

"""
#load files
print("loading nlcd data")
nlcd = rxr.open_rasterio("../raw_data/nlcd_raw/nlcd_clipped.tif", masked=True).squeeze()
print("loacing srtm data")
srtm = rxr.open_rasterio("../raw_data/srtm_raw/combined.tif", masked=True).squeeze()

    ###     ###     ###     ###
srtm_g = gdal.Open("../raw_data/srtm_raw/combined.tif", gdal.GA_ReadOnly)
print(type(srtm_g))
ox, pw, xskew, oy, yskew, ph = srtm_g.GetGeoTransform()
nd_value = 100000000
arr = srtm_g.ReadAsArray()

del srtm_g#, lyr

print("loading gedi data")
#gedi_pts = gpd.read_file("GEDI_2B_clean/GEDI_2B_clean.shp")
gedi_pts = shapefile.Reader("../raw_data/gedi_pts/GEDI_2B_clean.shp")
#print(type(gedi_pts))
print("done")
#print(gedi_pts.shapeTypeName)
print(len(gedi_pts), "points")
for f in range(len(gedi_pts.fields)):
    print(f, gedi_pts.fields[f])
grecs = gedi_pts.shapeRecords()

cut = len(gedi_pts)
print("running on first", cut, "points")
nlcd_trans = Transformer.from_crs("EPSG:4326", nlcd.rio.crs, always_xy=True)
nlcd_vals = []
srtm_elev = []
print("getting nlcd values")
print(arr.shape)
aspectinfo = np.zeros((arr.shape[0], arr.shape[1], len(filters)))
for i in range(cut): #len(gedi_pts)):
    a, b = [grecs[i].record[2], grecs[i].record[3]]
    print(a, b)
    #print()
    break
    xp, yp = nlcd_trans.transform(b, a)
    nlcd_vals.append(float(nlcd.sel(x=xp,y=yp, method='nearest').values))
    srtm_elev.append(float(srtm.sel(x=xp,y=yp, method='nearest').values))

    ##get coords, then neighbors coords
    #height. width?
    if True:
        ix, jy = get_index(xp, yp, ox, oy, pw, ph)
        fresults = []
        for f in filters:
            fresults.append(0)
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    nx = ix + j
                    ny = jy + k
                    if j == 0 and k == 0:
                        #check
                        if not nlcd_vals[-1] == float(nlcd.sel(x=inv_index(ix,jy,ox,oy,pw,ph)[0], y=inv_index(ix,jy,ox,oy,pw,ph)[1], method='nearest').values):
                            print("uhoh", xp, yp, " -> ", inv_index(ix, jy, ox, oy, pw, ph))
                    if nx >= 0 and nx < arr.shape[0] and ny >= 0 and ny < arr.shape[1]:
                        fresults[-1] += f[j+1, k+1]*arr[nx, ny]
        for ff in range(len(fresults)):
            aspectinfo[ix, jy, ff] = fresults[ff]
    #if str(nlcd_vals[-1]) != "nan":
    #    print(nlcd_vals[-1])
    if i % 50000 == 0:
        print(i)
#print(latlon)
print("making graphs")
print(nlcd_vals[:20])

plt.figure()
plt.hist(nlcd_vals)
plt.title("NLCD - land covers at GEDI data points")
plt.savefig("../figures/gedi_distributions/land_cover_dist.png")
plt.show()
plt.cla()
plt.close()

plt.figure()
plt.hist(srtm_elev)
plt.title("SRTM - elevation at GEDI data points")
plt.savefig("../figures/gedi_distributions/elevation_dist.png")
plt.show()
plt.close()
print("all done")

nlcd_val_idx = {}
nlcd_graph = []
nlcd_freqs = []
for i in range(cut):#len(nlcd_vals)):
    print(type(nlcd_vals[i]))
    if nlcd_vals[i] in nlcd_graph:
        nlcd_freqs[nlcd_val_idx[nlcd_vals[i]]] += 1
    else:
        nlcd_val_idx[nlcd_vals[i]] = len(nlcd_graph)
        nlcd_graph.append(nlcd_vals[i])
        nlcd_freqs.append(1)
#do graph
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(nlcd_graph, nlcd_freqs)
plt.show()
"""
