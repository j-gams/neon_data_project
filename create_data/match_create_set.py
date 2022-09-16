### Written by Jerry Gammie @j-gams

### WHAT DOES THIS CODE DO?
### - 


### command line arguments
### X raster data path(s)       [file path(s), comma separated, required]   [eg srtm, nlcd]
###                             - raster data to combine into X samples
### point interpol. data path   [file path(s), comma separated, required]   [eg gedi centroids]
###                             - shapefile of points with data fields to interpolate to
###                               spatial data
### y raster data path          [file path, required]                       [eg ecostress]
###                             - raster data to use as y value for samples
###                               X data will be cut to regions the size of y datas pixels
### y resolution                [int, required]                             [eg 70 for ecos]
###                             - resolution in meters of y raster data
### y pixel size                [int, required - res. of output data]       [eg 5]
###                             - resolution in meters of X samples created in this process
### create file structure       [T/F, required]
###                             - create a new dataset, or work in existing file structure
### file structure name         [file path, required]
###                             - path to root directory of dataset
### lo-memory mode              [--lomem or blank (defaults to high memory mode if blank), opt.]
###                             - whether to be cautious about loading a lot of data into memory
###                               at once. If in lo-memory mode the program will write relevant
###                               fields (specified by critical fields argument), if generate coordinates and/or
###                               generate other data is set to true, to individual
###                               files so that they can be loaded in individually as np arrays
###                             - every field containing the same keyword will be written to the
###                               same file
###                             - THIS WILL TAKE A LONG TIME but it might be necessary for RAM
###                             - if gencoords are false but lo-memory mode is true then it will attempt to load
###                               precomputed individual files
### generate coordinates        [--gencoords or blank (defaults to false if blank), optional]
###                             - generate a file with the lat/long to make finding neighbors
###                               a lot less memory intensive
### generate other data         [--genetc or blank (defaults to false if blank) optional]
###                             - generate critical field files
### override restructuring      [--override or blank (defaults to false if blank), optional]
###                             - whether to override pre-existing restructured data files
### skip save                   [--skipsave]
### hdf5 mode                   [--h5mode]
### hdf5 write both             [--h5both]
### no shuffle mode             [--noshuffle]
### prescreen extreme samples   [--prescreen]
### critical fields             [-c comma separated field keywords, optional]
###                             - fields to include in the output samples
### k closest approximation     [-k int, optional (default 10)]
### test mode                   [-t int, optional]
### channel mode                [-m {hwc, chw}]
###                             - whether to have channels as the first or third axis
### pad image                   [-p int]
###                             - whether to add pixels around the basic image (for nicer img dimensions?)
### pad hash                    [-h int]
###                             - how many gridsquares to add around the raster (to catch everything)
### h5 chunk size               [-u int, chunksize]
###
### set numpy random seed       [-s int, random seed]
### verbosity                   [-q {0, 1, 2}, optional (default 2, verbose)

### Usage Example:
### python match_create_set.py ../raw_data/srtm_raw/srtm_clipped.tif,../raw_data/nlcd_raw/nlcd_clipped.tif ../raw_data/gedi_pts/GEDI_2B_clean.shp ../raw_data/ecos_wue/WUE_Median_Composite_AOI.tif 70 5 true ../data/data_interpolated --lomem --gencoords --override --genetc -c cover,pavd,fhd -q 2
import sys
init_ok = True
x_raster_locs = []
point_shape = []
y_raster_loc = ""
y_res = 70
y_pix = 5
create_fs = True
fs_loc = "../data/default"
lo_mem = True
gen_coords = False
gen_etc = False
override_regen = False
critical_fields = []
verbosity = 2
k_approx = 10
testmode = -1
channel_first = False
pad_img = 0
hash_pad = 1
skip_save = False
h5_mode = False
h5_scsv = False
h5chunksize = 1000
shuffleorder=True
prescreen2=False
npseed = None
if len(sys.argv) < 8:
    init_ok = False
else:
    x_raster_locs = sys.argv[1].split(",")
    point_shape = sys.argv[2].split(",")
    y_raster_loc = sys.argv[3]
    y_res = int(sys.argv[4])
    y_pix = int(sys.argv[5])
    if sys.argv[6] == "T" or sys.argv[6] == 't' or sys.argv[6] == "True" or sys.argv[6] == "true":
        create_fs = True
    if sys.argv[6] == "F" or sys.argv[6] == 'f' or sys.argv[6] == "False" or sys.argv[6] == "false":
        create_fs = False
    fs_loc = sys.argv[7]
    for i in range(8, len(sys.argv)):
        if sys.argv[i] == "--lomem":
            lo_mem = True
        elif sys.argv[i] == "--gencoords":
            gen_coords = True
        elif sys.argv[i] == "--genetc":
            gen_etc = True
        elif sys.argv[i] == "--override":
            override_regen = True
        elif sys.argv[i] == "--skipsave":
            skip_save = True
        elif sys.argv[i] == "--h5mode":
            h5_mode = True
            h5_scsv = False
        elif sys.argv[i] == "--h5both":
            h5_mode = True
            h5_scsv = True
        elif sys.argv[i] == "--noshuffle":
            shuffleorder = False
        elif sys.argv[i] == "--prescreen":
            prescreen2 = True
        elif sys.argv[i] == "-c":
            critical_fields = sys.argv[i+1].split(",")
        elif sys.argv[i] == "-q":
            verbosity = int(sys.argv[i+1])
        elif sys.argv[i] == "-k":
            k_approx = int(sys.argv[i + 1])
        elif sys.argv[i] == "-t":
            testmode = int(sys.argv[i + 1])
        elif sys.argv[i] == "-m":
            if sys.argv[i+1] == "chw":
                channel_first = True
            else:
                channel_first = False
        elif sys.argv[i] == "-p":
            pad_img = int(sys.argv[i+1])
        elif sys.argv[i] == "-h":
            hash_pad = int(sys.argv[i+1])
        elif sys.argv[i] == "-u":
            h5chunksize = int(sys.argv[i+1])
        elif sys.argv[i] == "-s":
            np.random.seed(int(sys.argv[i+1]))
            npseed = int(sys.argv[i+1])

### TODO -- create a log of parameters used
#txt_out = open("../ " ".txt", "w+")
#log_string = ["* DATASET CREATED WITH THE FOLLOWING PARAMETERS"]


print("numpy random seed set to", npseed)
if not init_ok:
    sys.exit("missing or incorrect command line arguments")
imgsize = y_res // y_pix

def qprint(instr, power):
    if power <= verbosity:
        print(instr)

qprint("running in "+str(imgsize + pad_img*2)+" * "+str(imgsize + pad_img*2)+" mode (padding set to "+str(pad_img)+")",2)

qprint("importing packages", 2)
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr
#import geopandas as gpd
#import earthpy as et
#import earthpy.plot as ep
from osgeo import gdal
from osgeo import ogr
import shapefile
import h5py
#from longsgis import voronoiDiagram4plg

from rfdata_loader import rfloader, piloader

qprint("done importing packages", 2)
if channel_first:
    qprint("running in channel-first mode (chw)", 2)
else:
    qprint("running in channel-third mode (hwc)", 2)

found_coord = False
found_etc = [False for i in range(len(critical_fields))]
### REFACTORED CODE
if lo_mem:
    if gen_coords:
        qprint("generating restructured coordinate file", 2)
    else:
        qprint("not restructuring coordinates", 2)
    if gen_etc:
        qprint("generating restructured files for critical fields", 2)
    else:
        qprint("not restructuring data fields", 2)
if create_fs:
    qprint("checking for existing data file structure", 2)
    if not os.path.isdir(fs_loc):
        qprint("creating data file structure", 2)
        os.mkdir(fs_loc)
        os.mkdir(fs_loc + "/datasrc")
        os.mkdir(fs_loc + "/datasrc/x_img")
        os.mkdir(fs_loc + "/point_reformat")
        os.mkdir(fs_loc + "/meta")
    else:
        qprint("file structure already exists, skipping", 2)
    if lo_mem:
        if gen_coords:
            if override_regen:
                qprint("forcibly creating file for reformatted coordinate data", 2)
                os.system("rm " + fs_loc + "/point_reformat/geo_coords.txt")
                os.system("touch " + fs_loc + "/point_reformat/geo_coords.txt")
            elif not os.path.exists(fs_loc + "/point_reformat/geo_coords.txt"):
                qprint("no file detected, creating file for reformatted coordinate data", 2)
                os.system("touch " + fs_loc + "/point_reformat/geo_coords.txt")
            else:
                qprint("reformatted coordinate file detected, skipping.", 2)
                found_coord = True
        if gen_etc:
            for i in range(len(critical_fields)):
                if override_regen:
                    qprint("forcibly creating file for reformatted " + critical_fields[i] + " data", 2)
                    os.system("rm " + fs_loc + "/point_reformat/pt_" + critical_fields[i] + ".txt")
                    os.system("touch " + fs_loc + "/point_reformat/pt_" + critical_fields[i] + ".txt")
                elif not os.path.exists(fs_loc + "/point_reformat/pt_" + critical_fields[i] + ".txt"):
                    qprint("no file detected, creating file for reformatted " + critical_fields[i] + " data", 2)
                    os.system("touch " + fs_loc + "/point_reformat/pt_" + critical_fields[i] + ".txt")
                else:
                    qprint("reformatted " + critical_fields[i] + " file detected, skipping.", 2)
                    found_etc[i] = True

    test_img_ = np.zeros((3, 14, 14))
    arrReshaped = test_img_.reshape(test_img_.shape[0], -1) #see bookmarked page on how to invert this
    np.savetxt(fs_loc + "/datasrc/x_img/x_test.csv", arrReshaped, delimiter=",", newline="\n")
    print("shape:", arrReshaped.shape)

### load x raster
xraster = []
layernames = []
ndv_vals = []
xr_rsize = []
xr_params = []
xr_npar = []

qprint("loading raster data", 1)
for loc in x_raster_locs:
    tdataname = loc.split("/")[-1]
    qprint("loading " + tdataname + " data...", 2)
    xraster.append(gdal.Open(loc))
    layernames.append(tdataname.split(".")[0])
    tdata_rband = xraster[-1].GetRasterBand(1)
    ndv_vals.append(tdata_rband.GetNoDataValue())
    xr_rsize.append((xraster[-1].RasterXSize, xraster[-1].RasterYSize))
    tulh, tpxh, _, tulv, _, tpxv = xraster[-1].GetGeoTransform()
    tpxv = abs(tpxv)
    ###print crs info
    xr_params.append((tulh, tulv, tpxh, tpxv))
    xr_npar.append(xraster[-1].ReadAsArray().transpose())

### load y raster
tyname = y_raster_loc.split("/")[-1]
qprint("loading " + tyname + " data...", 2)
yraster = gdal.Open(y_raster_loc)
yrband = yraster.GetRasterBand(1)
yndv = yrband.GetNoDataValue()
print("no data value", yndv)
yrsize = (yraster.RasterXSize, yraster.RasterYSize)
qprint("y raster dimensions: " + str(yrsize), 2)
yulh, ypxh, _, yulv, _, ypxv =yraster.GetGeoTransform()
ypxv = abs(ypxv)
### print crs info
print(yulh, yulv, ypxh, ypxv)
y_npar = yraster.ReadAsArray().transpose()


xpoints = []
ptlayers = []
ptindexer = {}
grecs = []
ptlnames = []
qprint("loading shape-point data", 1)
for pt in point_shape:
    if not lo_mem or (gen_coords or gen_etc): #or True:
        tdataname = pt.split("/")[-1]
        qprint("loading " + tdataname + " data...", 2)
        xpoints.append(shapefile.Reader(pt))
        qprint("loading " + tdataname + " records...", 2)
        grecs.append(xpoints[-1].shapeRecords())

        ### TODO - STORE PTLAYERS, PTINDEXER to avoid this every damn time
        for i in range(len(critical_fields)):
            ptl_idx = 0
            for j in range(len(xpoints[-1].fields)):
                if critical_fields[i] in xpoints[-1].fields[j][0]:
                    ### given the layer (key) provides the ptfile #, critical field #, field index within that shape,
                    ### and id within the np array (if exists)
                    ptindexer[len(ptlayers)] = (len(xpoints) -1, i, j, ptl_idx)
                    ptlayers.append(critical_fields[i] + "_" + str(ptl_idx))
                    ptlnames.append(xpoints[-1].fields[j][0])
                    ptl_idx += 1

        ### save
        print("writing point indexer file")
        if os.path.exists(fs_loc + "/point_reformat/pt_indexer_util.txt"):
            os.system("rm " + fs_loc + "/point_reformat/pt_indexer_util.txt")
        for lkey in ptindexer.keys():
            lkstr = ""
            for elt in ptindexer[lkey]:
                lkstr += str(elt) + ","
            os.system('echo "' + str(lkey) + ':' + lkstr[:-1] + '" >> ' + fs_loc + '/point_reformat/pt_indexer_util.txt')

    if lo_mem:
        if gen_coords and not found_coord:
            qprint("reformatting shapefile coordinates. This could take a while...", 1)
            if verbosity > 0:
                print("progress 0/50 ", end="", flush=True)
            progress = 0
            os.system('echo "lon,lat" >> ' + fs_loc + '/point_reformat/geo_coords.txt')
            for i in range(len(grecs[-1])):
                progress += 1
                if verbosity > 0 and progress % (len(grecs[-1]) // 50) == 0:
                    print("-", end="", flush=True)
                a, b = grecs[-1][i].record[3], grecs[-1][i].record[2]
                os.system('echo "' + str(a) + ',' + str(b) + '" >> ' + fs_loc + '/point_reformat/geo_coords.txt')

            qprint("---> 50/50 (done)", 1)

        for i in range(len(critical_fields)):
            if gen_etc and not found_etc[i]:
                qprint("reformatting critical field " + critical_fields[i] + ". This could take a while...", 1)
                if verbosity > 0:
                    print("progress 0/50 ", end="", flush=True)
                progress = 0
                ### build list of field indices, names:
                fields_ids = []
                fields_names = ""
                for k in range(len(xpoints[-1].fields)):
                    if critical_fields[i] in xpoints[-1].fields[k][0]:
                        fields_ids.append(k)
                        fields_names += xpoints[-1].fields[k][0] + ","
                ### header
                os.system('echo "' + fields_names[:-1] + '" >> ' + fs_loc + '/point_reformat/pt_' + critical_fields[i] + '.txt')
                for j in range(len(grecs[-1])):
                    progress += 1
                    if verbosity > 0 and progress % (len(grecs[-1]) // 50) == 0:
                        print("-", end="", flush=True)
                    dumpstr = ""
                    for k in fields_ids:
                    #for k in range(len(xpoints[-1].fields)):
                        if critical_fields[i] in xpoints[-1].fields[k][0]:
                            dumpstr += str(grecs[-1][j].record[k]) + ","
                    os.system('echo "' + dumpstr[:-1] + '" >> ' + fs_loc + '/point_reformat/pt_' + critical_fields[i] + '.txt')
                print("---> 50/50 (done)")
                print("done reformatting " + critical_fields[i] + " data")
        if not lo_mem or (gen_coords or gen_etc):
            del xpoints[-1]
            del grecs[-1]

print("layer number audit: ", len(ptlayers))

### SAVE CHANNEL NAMES
#save_channelnames = layernames + ptlnames
#os.system("rm " + fs_loc + "/meta/channel_names.txt")
#os.system("touch " + fs_loc + "/meta/channel_names.txt")
#for cname in save_channelnames:
#    os.system('echo "' + cname + '," >> ' + fs_loc + '/meta/channel_names.txt')

ptlayers = []
### now try to load the data back in
if lo_mem:
    ### clear hi-mem data (moved to above)
    print("still in lo-memory mode")
    npcoords, _ = rfloader(fs_loc + '/point_reformat/geo_coords.txt')
    #npcoords = np.genfromtxt(fs_loc + '/point_reformat/geo_coords.txt', delimiter=',')
    print(npcoords.shape)
    ### load data from file

    crit_npar = []
    print("nan values:")
    print(np.count_nonzero(np.isnan(npcoords)))

    ###load ptindexer
    print("loading ptindexer")
    ptindexer = piloader(fs_loc + "/point_reformat/pt_indexer_util.txt")
    for i in range(len(critical_fields)):
        ##load in
        loaddata, fnames = rfloader(fs_loc + '/point_reformat/pt_' + critical_fields[i] + '.txt')
        #crit_npar.append(rfloader(fs_loc + '/point_reformat/pt_' + critical_fields[i] + '.txt'))
        crit_npar.append(loaddata)
        ptlayers += fnames
        del loaddata

        #crit_npar.append(np.genfromtxt(fs_loc + '/point_reformat/pt_' + critical_fields[i] + '.txt', delimiter=','))
        print(crit_npar[-1].shape)
        if crit_npar[-1].ndim < 2:
            print("reshaping")
            cnpdim = len(crit_npar[-1])
            print(cnpdim)
            crit_npar[-1] = np.reshape(crit_npar[-1], (-1, 1))
            print(crit_npar[-1].shape)
            print("np.unique", len(np.unique(crit_npar[-1])))
        print("nan values:")
        print(np.count_nonzero(np.isnan(crit_npar[-1])))
    print("layer number audit: ", len(ptlayers))
### now we have all the data loaded in?
### SAVE CHANNEL NAMES
save_channelnames = layernames + ptlayers
os.system("rm " + fs_loc + "/meta/channel_names.txt")
os.system("touch " + fs_loc + "/meta/channel_names.txt")
for cname in save_channelnames:
    os.system('echo "' + cname + '," >> ' + fs_loc + '/meta/channel_names.txt')

def cgetter(index, xy):
    if lo_mem:
        return npcoords[index, xy]
    else:
        return grecs[0][index].record[3-xy]
def clen():
    if lo_mem:
        return npcoords.shape[0]
    else:
        return len(grecs[0])

def pgetter(layer, index):
    #print(layer)
    if lo_mem:
        ### use crit_npar
        return crit_npar[ptindexer[layer][1]][index, ptindexer[layer][3]]
    else:
        return grecs[ptindexer[layer][0]][index].record[ptindexer[layer][2]]

# in geotif, raster data stored from upper left pixel coordinates, pixel width, and rotation.
# so we can get value at coordinate (x,y) from looking at pixel at
# (x - UL_x)/size_x
# (UL_y - y)/size_y
# here these are switched because in the geographic crs, increasing coordinates go up and to the right
# however within the matrix they go down and to the right, so the sign must be reversed for y

#turn coordinates into an index in the data array
def coords_idx(cx, cy, ulh, ulv, psh, psv):
    ix = int((cx - ulh)/psh)
    iy = int((ulv - cy)/psv)
    return ix, iy

# get coordinates of pixel from index
# if mode is 'ctr': get coords of center of pixel
# else if mode is 'ul': get coords of upper left of pixel
def idx_pixctr(ix, iy, ulh, ulv, psh, psv, mode='ul'):
    offsetx = 0
    offsety = 0
    if mode=='ctr':
        offsetx = psh/2
        offsety = psv/2
    cx = ulh + (ix * psh) + offsetx
    cy = ulv - (iy * psv) + offsety
    return cx, cy

def cdist(x1, y1, x2, y2):
    return (((x1 - x2) ** 2) + ((y1 - y2) ** 2))

### stupid k nearest
def getkclose(shapes, centerx, centery, k, ulh, ulv, psh, psv):
    distlist = []
    ids = []
    cx, cy = idx_pixctr(0.5+centerx, 0.5+centery, ulh, ulv, psh, psv, mode='ul')
    for i in range(len(shapes)):
        a = shapes[i, 0]
        b = shapes[i, 1]
        distlist.append(cdist(a, b, cx, cy))
        ids.append(i)
    #sort ids by distlist
    ids = [id for _, id in sorted(zip(distlist, ids), key=lambda pair:pair[0])]
    return ids[:k]

maxringsize = 0
avgringsize = 0

### hash all of the gedi footprints! (this could take a while lmao -- rip ram
print("doing the hash thing")
### create it with a buffer
ygrid_pt_hash = np.zeros((yrsize[0] + (2*hash_pad), yrsize[1] + (2*hash_pad)), dtype='object')
for i in range(ygrid_pt_hash.shape[0]):
    for j in range(ygrid_pt_hash.shape[1]):
        ygrid_pt_hash[i, j] = []
#ygrid_pt_hash = [list([list([]) for ii in range(yrsize[1] + 2)]) for jj in range(yrsize[0] + 2)]
#ygrid_pt_hash[0][0].append(5)
pstep = clen() // 50
print(clen())
actual_added = 0
for i in range(clen()):
    ### get coordinates
    if i % pstep == 0:
        print("-", end="", flush=True)
    xi, yi = coords_idx(cgetter(i, 0), cgetter(i, 1), yulh, yulv, ypxh, ypxv)
    if xi+hash_pad < 0 or yi+hash_pad < 0 or xi > yrsize[0]+(hash_pad*2) or yi > yrsize[1]+(2*hash_pad):
        print("big uh-oh!!!")
        print(i)
        print(cgetter(i, 0), cgetter(i, 1))
        print(xi, yi)
    else:
        actual_added += 1
        ygrid_pt_hash[xi+hash_pad, yi+hash_pad].append(i)
print("done doing the hash thing")
print("actually added", actual_added, "gedi points to hash (within bounds)")
#print(ygrid_pt_hash[1000:1010, 1000:1010])
"""for i in range(ygrid_pt_hash.shape[0]):
    for j in range(ygrid_pt_hash.shape[1]):
        if ygrid_pt_hash[i, j] != []:
            print(ygrid_pt_hash[i:i+10, j:j+10])"""

### TODO --- stop at sqrt(2) after the first one
### THE ROOT2 VERSION !!!
mem_root2 = math.sqrt(2)
def krings(x_in, y_in, min_k):
    ring_size = 0
    found_list = []
    cap = -1
    while(cap < 0 or ring_size <= cap):
        i_boundaries = [max(0-hash_pad, x_in-ring_size), min(yrsize[0]+hash_pad, x_in+ring_size+1)]
        j_boundaries = [max(0-hash_pad, y_in-ring_size), min(yrsize[1]+hash_pad, y_in+ring_size+1)]
        for i in range(i_boundaries[0], i_boundaries[1]):
            for j in range(j_boundaries[0], j_boundaries[1]):
                if i == i_boundaries[0] or i+1 == i_boundaries[1] or j == j_boundaries[0] or j+1 == j_boundaries[1]:
                    if len(ygrid_pt_hash[i+1, j+1]) > 0:
                        if cap == -1:
                            cap = max(math.ceil(mem_root2 * ring_size), 1) + 1
                        for k in ygrid_pt_hash[i+1, j+1]:
                            found_list.append(k)
        ring_size += 1
    return found_list, ring_size

def depricated_krings(x_in, y_in, min_k):
    ring_size = 0
    found_list = []
    while(len(found_list) < min_k):
        i_boundaries = [max(-1, x_in-ring_size), min(yrsize[0]+1, x_in+ring_size+1)]
        j_boundaries = [max(-1, y_in-ring_size), min(yrsize[1]+1, y_in+ring_size+1)]
        #print(i_boundaries, j_boundaries, x_in, y_in)
        for i in range(i_boundaries[0], i_boundaries[1]):
            for j in range(j_boundaries[0], j_boundaries[1]):
                if i == i_boundaries[0] or i + 1 == i_boundaries[1] or j == j_boundaries[0] or j + 1 == j_boundaries[1]:
                    for k in ygrid_pt_hash[i+1, j+1]:
                    #if i in i_boundaries or j in j_boundaries or i+1 in i_boundaries or j+1 in j_boundaries:
                        found_list.append(k)
                        #print("got one")
        #print(ring_size)
        ring_size += 1
    return found_list, ring_size

### ok now actually build the data
#if testmode > 0:
#    pr_unit = testmode // 50
#else:
### TODO --- make this 16x16 instead of 14x14... add 1 unit of buffer on each side
pr_unit = (yrsize[0] * yrsize[1]) // 50
qprint("each step represents " + str(pr_unit) + " samples generated", 1)
progress = 0
nsuccess = 0
extreme_encounter = [0 for ii in range(len(xr_npar) + len(ptlayers) + 2)]
database = []
channels = len(xr_npar) + len(ptlayers) + 2
pd_colnames = ["filename", "y_value", "file_index", "yraster_x", "yraster_y", "avg_mid_dist"]
landmark_x, landmark_y = coords_idx(-104.876653,41.139535, yulh, yulv, ypxh, ypxv)
if skip_save:
    print("warning: running in skip save mode")

if h5_mode:
    #h5chunksize=1000
    print("running in h5 mode!")
    os.system("rm " + fs_loc + "/datasrc/x_h5.h5")
    h5_dataset = h5py.File(fs_loc + "/datasrc/x_h5.h5", "a")
    if channel_first:
        h5dset = h5_dataset.create_dataset("data", (h5chunksize, channels, imgsize + (2 * pad_img), imgsize + (2 * pad_img)),
                                           maxshape=(None, channels, imgsize+(2*pad_img), imgsize+(2*pad_img)),
                                           chunks=(h5chunksize,channels, imgsize+(2*pad_img), imgsize+(2*pad_img)))
    else:
        h5dset = h5_dataset.create_dataset("data", (h5chunksize, imgsize+(2*pad_img), imgsize+(2*pad_img), channels),
                                           maxshape=(None,imgsize+(2*pad_img), imgsize+(2*pad_img), channels),
                                           chunks=(h5chunksize,imgsize+(2*pad_img), imgsize+(2*pad_img), channels))
    h5len = 0
    h5tid = 0
    h5chunkid = 0
    if channel_first:
        h5_chunk = np.zeros((h5chunksize, channels, imgsize + (2 * pad_img), imgsize + (2 * pad_img)))
    else:
        h5_chunk = np.zeros((h5chunksize, imgsize + (2 * pad_img), imgsize + (2 * pad_img), channels))
        h5_chunk.fill(-1)

diids = [ii for ii in range(101)]
dists = [0 for ii in range(101)]

dbins = [[0 for jj in range(20)] for ii in range(len(xr_npar))]
dbinsmins = [0, 0, 0, 0, 0]
dbinsmaxs = [4500, 260, 70, 360, 3.5]

#if verbosity > 0:
#    print("progress 0/50 ", end="", flush=True)

prescreen1=False
extreme_bounds = [-100000, 100000]
prescreen_dist = 35 ** 2
prescreen_forestp = 0.95
if shuffleorder:
    irange_default = np.arange(yrsize[0])
    jrange_default = np.arange(yrsize[1])
    np.random.shuffle(irange_default)
    np.random.shuffle(jrange_default)
else:
    irange_default = np.arange(yrsize[0])
    jrange_default = np.arange(yrsize[1])

for i in irange_default:
    for j in jrange_default:
        progress += 1
        temp_lcpercent = 0
        #failsafe_copy = list(dbins)
        #if verbosity > 0 and progress % pr_unit == 0:
        #    print("-", end="", flush=True)
        extreme_warning = False
        if y_npar[i, j] != yndv:
            if not h5_mode or (h5_mode and h5_scsv):
                if channel_first:
                    x_img = np.zeros((channels, imgsize+(2*pad_img), imgsize+(2*pad_img)))
                else:
                    x_img = np.zeros((imgsize+(2*pad_img), imgsize+(2*pad_img), channels))
            nlcd_count = 0
            for k in range(len(xr_npar)):# in xr_npar:
                #binify = dbinsmaxs[k] - dbinsmins[k]
                ### ... Try again with a buffer to get 16x16 image
                for si in range(0 - pad_img, imgsize+pad_img):
                    for sj in range(0 - pad_img, imgsize+pad_img):
                        ### want -.5, .5, 1.5, 2.5, etc...
                        sxoffset = ((2 * si) + 1) / (2 * imgsize)
                        syoffset = ((2 * sj) + 1) / (2 * imgsize)
                        tempx, tempy = idx_pixctr(i + sxoffset, j + syoffset, yulh, yulv, ypxh,
                                ypxv, mode = 'ul')
                        tempi, tempj = coords_idx(tempx, tempy, xr_params[k][0], xr_params[k][1],
                                xr_params[k][2], xr_params[k][3])
                        #...
                        ### extreme encounters
                        if  xr_npar[k][tempi, tempj] > extreme_bounds[1] or xr_npar[k][tempi, tempj] < extreme_bounds[0]:
                            ### extreme
                            extreme_warning = True
                            extreme_encounter[k] += 1
                        if k == 1 and (xr_npar[k][tempi, tempj] <40 or xr_npar[k][tempi, tempj] > 45):
                            nlcd_count += 1
                        if not h5_mode or (h5_mode and h5_scsv):
                            if channel_first:
                                x_img[k, si+pad_img, sj+pad_img] = xr_npar[k][tempi, tempj]
                            else:
                                x_img[si+pad_img, sj+pad_img, k] = xr_npar[k][tempi, tempj]
                        if h5_mode:
                            if channel_first:
                                h5_chunk[h5tid, k, si+pad_img, sj+pad_img] = xr_npar[k][tempi, tempj]
                            else:
                                h5_chunk[h5tid, si+pad_img, sj+pad_img, k] = xr_npar[k][tempi, tempj]
                        ### do binning for analysis
            #non-forest fraction of x
            #temp_lcpercent = nlcd_count / ((imgsize + (2*pad_img)) ** 2)
            #dists[round((((imgsize + (2*pad_img)) ** 2) - nlcd_count)/((imgsize + (2*pad_img)) ** 2) * 100)] += 1
            k_ids, rings = krings(i, j, k_approx)
            avgringsize += rings
            if rings > maxringsize:
                maxringsize = rings
            ### ... buffered (16x16) trial
            for si in range(0-pad_img, imgsize+pad_img):
                for sj in range(0-pad_img, imgsize+pad_img):
                    sxoffset = ((2 * si) + 1) / (2 * imgsize)
                    syoffset = ((2 * sj) + 1) / (2 * imgsize)
                    tempx, tempy = idx_pixctr(i + sxoffset, j + syoffset, yulh, yulv, ypxh,
                            ypxv, mode='ul')
                    mindist = 1000000
                    minpt = None
                    for pt_idx in k_ids:
                        tdist = cdist(npcoords[pt_idx, 0], npcoords[pt_idx, 1], tempx, tempy)
                        if tdist < mindist:
                            mindist = tdist
                            minpt = pt_idx
                    for m in range(len(ptlayers)):
                        if not h5_mode or (h5_mode and h5_scsv):
                            if channel_first:
                                x_img[len(xr_npar) + m, si+pad_img, sj+pad_img] = pgetter(m, minpt)
                            else:
                                x_img[si+pad_img, sj+pad_img, len(xr_npar) + m] = pgetter(m, minpt)
                        if h5_mode:
                            if channel_first:
                                h5_chunk[h5tid, len(xr_npar)+m, si+pad_img, sj+pad_img] = pgetter(m, minpt)
                            else:
                                h5_chunk[h5tid, si+pad_img, sj+pad_img, len(xr_npar)+m] = pgetter(m, minpt)
                        if pgetter(m, minpt) > 10000 or pgetter(m, minpt) < -10000:
                            extreme_encounter[m + len(xr_npar)] += 1
                    if not h5_mode or (h5_mode and h5_scsv):
                        if channel_first:
                            x_img[len(xr_npar) + len(ptlayers), si+pad_img, sj+pad_img] = minpt
                            x_img[len(xr_npar) + len(ptlayers) + 1, si+pad_img, sj+pad_img] = mindist
                        else:
                            x_img[si+pad_img, sj+pad_img, len(xr_npar) + len(ptlayers)] = minpt
                            x_img[si+pad_img, sj+pad_img, len(xr_npar) + len(ptlayers) + 1] = mindist
                    if h5_mode:
                        if channel_first:
                            h5_chunk[h5tid, len(xr_npar)+len(ptlayers), si+pad_img, + sj+pad_img] = minpt
                            h5_chunk[h5tid, len(xr_npar)+len(ptlayers) + 1, si+pad_img, + sj+pad_img] = mindist
                        else:
                            h5_chunk[h5tid, si+pad_img, sj+pad_img, len(xr_npar)+len(ptlayers)] = minpt
                            h5_chunk[h5tid, si+pad_img, sj+pad_img, len(xr_npar)+len(ptlayers) + 1] = mindist
            if h5_mode:
                if channel_first:
                    avg_mid_dist = h5_chunk[h5tid, -1, (imgsize + (pad_img * 2)) // 2, (imgsize + (pad_img * 2)) // 2] / 4
                    avg_mid_dist += h5_chunk[h5tid, -1, (imgsize + (pad_img * 2) - 1) // 2, (imgsize + (pad_img * 2)) // 2] / 4
                    avg_mid_dist += h5_chunk[h5tid, -1, (imgsize + (pad_img * 2)) // 2, (imgsize + (pad_img * 2) - 1) // 2] / 4
                    avg_mid_dist += h5_chunk[h5tid, -1, (imgsize + (pad_img * 2) - 1) // 2, (imgsize + (pad_img * 2) - 1) // 2] / 4
                else:
                    avg_mid_dist = h5_chunk[h5tid, (imgsize + (pad_img * 2)) // 2, (imgsize + (pad_img * 2)) // 2, -1] / 4
                    avg_mid_dist += h5_chunk[h5tid, (imgsize + (pad_img * 2) - 1) // 2, (imgsize + (pad_img * 2)) // 2, -1] / 4
                    avg_mid_dist += h5_chunk[h5tid, (imgsize + (pad_img * 2)) // 2, (imgsize + (pad_img * 2) - 1) // 2, -1] / 4
                    avg_mid_dist += h5_chunk[h5tid, (imgsize + (pad_img * 2) - 1) // 2, (imgsize + (pad_img * 2) - 1) // 2, -1] / 4
            else:
                if channel_first:
                    avg_mid_dist =  x_img[-1, (imgsize+(pad_img * 2))//2, (imgsize+(pad_img * 2))//2]/4
                    avg_mid_dist += x_img[-1, (imgsize+(pad_img * 2) - 1)//2, (imgsize + (pad_img*2))//2]/4
                    avg_mid_dist += x_img[-1, (imgsize+(pad_img * 2))//2, (imgsize + (pad_img * 2) - 1)//2]/4
                    avg_mid_dist += x_img[-1, (imgsize+(pad_img * 2) - 1)//2, (imgsize + (pad_img * 2) - 1)//2]/4
                else:
                    avg_mid_dist =  x_img[(imgsize + (pad_img * 2))//2, (imgsize + (pad_img * 2))//2, -1]/4
                    avg_mid_dist += x_img[(imgsize + (pad_img * 2) - 1)//2, (imgsize + (pad_img * 2))//2, -1]/4
                    avg_mid_dist += x_img[(imgsize + (pad_img * 2))//2, (imgsize + (pad_img * 2) - 1)//2, -1]/4
                    avg_mid_dist += x_img[(imgsize + (pad_img * 2) - 1)//2, (imgsize + (pad_img * 2) - 1)//2, -1]/4
            
            ### make a string of
            ### file name, y value, nsuccess, y raster coordinates, ..., average distance to nearest neighbor
            if prescreen1:
                if avg_mid_dist <= prescreen_dist and temp_lcpercent <= (1 - prescreen_forestp):
                    #good to save
                    if not skip_save:
                        database.append(["/datasrc/x_img/x_" +str(nsuccess)+ ".csv", y_npar[i, j], nsuccess, i, j, avg_mid_dist])
                        if channel_first:
                            np.savetxt(fs_loc + "/datasrc/x_img/x_" +str(nsuccess)+ ".csv", x_img.reshape(x_img.shape[0], -1),
                                    delimiter=",", newline="\n")
                        else:
                            np.savetxt(fs_loc + "/datasrc/x_img/x_" +str(nsuccess)+ ".csv", x_img.reshape(-1, x_img.shape[2]),
                                    delimiter=",", newline="\n")
                    nsuccess += 1
                    if verbosity > 0 and nsuccess % (testmode // 50) == 0:
                        print("-", end="", flush=True)
                #else:
                #    dbins = list(failsafe_copy)
            elif (not extreme_warning and prescreen2) or not prescreen2:
                if not skip_save and (not h5_mode or (h5_mode and h5_scsv)):
                    #database.append(["/datasrc/x_img/x_" +str(nsuccess)+ ".csv", y_npar[i, j], nsuccess, i, j, avg_mid_dist])
                    if channel_first:
                        np.savetxt(fs_loc + "/datasrc/x_img/x_" +str(nsuccess)+ ".csv", x_img.reshape(x_img.shape[0], -1),
                                delimiter=",", newline="\n")
                    else:
                        np.savetxt(fs_loc + "/datasrc/x_img/x_" +str(nsuccess)+ ".csv", x_img.reshape(-1, x_img.shape[2]),
                                delimiter=",", newline="\n")
                if not skip_save and h5_mode:
                    #database.append(["/datasrc/x_img/x_" + str(nsuccess) + ".csv", y_npar[i, j], nsuccess, i, j, avg_mid_dist])
                    h5tid += 1
                    h5len += 1
                    #if (h5len + 1) % h5chunksize == 0:
                    if h5tid % (h5chunksize//10) == 0:
                        print("-", end="", flush=True)
                    if h5tid == h5chunksize:
                        h5chunkid += 1
                        print("> resizing h5 (" + str(h5chunkid) + ")")
                        h5dset.resize(h5len, axis=0)
                        h5dset[h5len-h5chunksize:h5len,:,:,:] = np.array(h5_chunk[:,:,:,:])
                        h5_chunk = np.zeros(h5_chunk.shape)
                        h5_chunk.fill(-1)
                        h5tid = 0
                if not skip_save:
                    database.append(["/datasrc/x_img/x_" + str(nsuccess) + ".csv", y_npar[i, j], nsuccess, i, j, avg_mid_dist])
                nsuccess += 1
            if testmode > 0 and nsuccess > testmode:
                print()
                print("max ring size: ", maxringsize)
                print("avg ring size: ", avgringsize//nsuccess)
                print("saving ydata")
                ydataframe = pd.DataFrame(data=database, columns=pd_colnames)
                ydataframe.to_csv(fs_loc + "/datasrc/ydata.csv")
                #save h5set
                if h5_mode:
                    ###need to make sure last chunk is saved
                    print("saving last h5 chunk...")
                    h5dset.resize(h5len, axis=0)
                    h5dset[h5len-h5tid:h5len,:,:,:] = h5_chunk[:h5tid,:,:,:]
                    print("saving h5 dset...")
                    h5_dataset.close()
                #print("max ring size: ", maxringsize)
                print("extreme encounter report:")
                no_enc = True
                for i in range(len(extreme_encounter)):
                    if extreme_encounter[i] > 0:
                        no_enc = False
                        print(" ", i, extreme_encounter[i])
                if no_enc:
                    print("no extreme encounters")

                plt.figure()
                plt.bar(diids, dists)
                plt.title("distribution of nlcd values over 5m regions / sample")
                plt.savefig("../figures/nlcd_dist_.png")
                plt.cla()
                plt.close()

                for i in range(len(dists)):
                    if dists[i] != 0:
                        dists[i] = math.log(dists[i])
                plt.figure()
                plt.bar(diids, dists)
                plt.title("log distribution of nlcd values over 5m regions / sample")
                plt.savefig("../figures/nlcd_dist_log.png")
                plt.cla()
                plt.close()

                ### dist things
                #rasters_names_list = ["srtm", "nlcd", "slope", "aspect", "ecostress_WUE"]
                #for idb in range(len(dbins)):
                #    plt.figure()
                #    plt.bar(dbins[idb])
                #    plt.title(rasters_names_list[idb] + " distribution over 5m pixels")
                #    plt.savefig("../figures/pixel_distributions/" + rasters_names_list[idb] + "_dbn.png")
                #    plt.cla()
                #    plt.close()

                sys.exit("exiting after testmode samples")

print()
#print(maxringsize)
"""
print("max ring size: ", maxringsize)
print("saving ydata")
ydataframe = pd.DataFrame(data=database, columns=pd_colnames)
ydataframe.to_csv(fs_loc + "/datasrc/ydata.csv")"""

print("max ring size: ", maxringsize)
print("avg ring size: ", avgringsize//nsuccess)
print("saving ydata")
ydataframe = pd.DataFrame(data=database, columns=pd_colnames)
ydataframe.to_csv(fs_loc + "/datasrc/ydata.csv")
#save h5set
if h5_mode:
    ###need to make sure last chunk is saved
    print("saving last h5 chunk...")
    h5dset.resize(h5len, axis=0)
    h5dset[h5len-h5tid:h5len,:,:,:] = h5_chunk[:h5tid,:,:,:]
    print("saving h5 dset...")
    h5_dataset.close()
#print("max ring size: ", maxringsize)
print("extreme encounter report:")
no_enc = True
for i in range(len(extreme_encounter)):
    if extreme_encounter[i] > 0:
        no_enc = False
        print(" ", i, extreme_encounter[i])
if no_enc:
    print("no extreme encounters")

plt.figure()
plt.bar(diids, dists)
plt.title("distribution of nlcd values over 5m regions / sample")
plt.savefig("../figures/nlcd_dist_.png")
plt.cla()
plt.close()

for i in range(len(dists)):
    if dists[i] != 0:
        dists[i] = math.log(dists[i])
plt.figure()
plt.bar(diids, dists)
plt.title("log distribution of nlcd values over 5m regions / sample")
plt.savefig("../figures/nlcd_dist_log.png")
plt.cla()
plt.close()