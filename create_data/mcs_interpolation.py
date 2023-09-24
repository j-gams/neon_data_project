### Written by Jerry Gammie @j-gams

### Usage Example:
### python match_create_set.py ../raw_data/srtm_raw/srtm_clipped.tif,../raw_data/nlcd_raw/nlcd_clipped.tif ../raw_data/gedi_pts/GEDI_2B_clean.shp ../raw_data/ecos_wue/WUE_Median_Composite_AOI.tif 70 5 true ../data/data_interpolated --lomem --gencoords --override --genetc -c cover,pavd,fhd -q 2
import sys

def isf(s):
    if s == "F" or s == "f" or s == "False" or s == "false":
        return True
    return False

def ist(s):
    if s == "T" or s == "t" or s == "True" or s == "true":
        return True
    return False


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
shuffleorder = True
prescreen2 = False
npseed = None
if len(sys.argv) < 8:
    init_ok = False
else:
    x_raster_locs = sys.argv[1].split(",")
    point_shape = sys.argv[2].split(",")
    y_raster_loc = sys.argv[3]
    y_res = int(sys.argv[4])
    y_pix = int(sys.argv[5])
    if ist(sys.argv[6]):
        create_fs = True
    if isf(sys.argv[6]):
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
            critical_fields = sys.argv[i + 1].split(",")
        elif sys.argv[i] == "-q":
            verbosity = int(sys.argv[i + 1])
        elif sys.argv[i] == "-k":
            k_approx = int(sys.argv[i + 1])
        elif sys.argv[i] == "-t":
            testmode = int(sys.argv[i + 1])
        elif sys.argv[i] == "-m":
            if sys.argv[i + 1] == "chw":
                channel_first = True
            else:
                channel_first = False
        elif sys.argv[i] == "-p":
            pad_img = int(sys.argv[i + 1])
        elif sys.argv[i] == "-h":
            hash_pad = int(sys.argv[i + 1])
        elif sys.argv[i] == "-u":
            h5chunksize = int(sys.argv[i + 1])
        elif sys.argv[i] == "-s":
            np.random.seed(int(sys.argv[i + 1]))
            npseed = int(sys.argv[i + 1])
    ### NEW PARSER
    for i in range(8, len(sys.argv)):
        ### T/F
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
        elif sys.argv[i] == "--noshuffle":
            shuffleorder = False
        elif sys.argv[i] == "--prescreen":
            prescreen2 = True
        ### Other Args
        elif sys.argv[i][:9] == "--h5mode=":
            if sys.argv[i][9:] == "h5":
                h5_mode = True
                h5_scsv = False
            elif sys.argv[i][9:] == "both":
                h5_mode = True
                h5_scsv = True
        elif sys.argv[i][:10] == "--cfields=":
            critical_fields = sys.argv[i][10:].split(",")
        elif sys.argv[i][:10] == "--kapprox=":
            k_approx = int(sys.argv[i][10:])
        elif sys.argv[i][:7] == "--test=":
            testmode = int(sys.argv[i][7:])
        elif sys.argv[i][:9] == "--orient=":
            if sys.argv[i][9:] == "chw":
                channel_first = True
            else:
                channel_first = False
        elif sys.argv[i][:6] == "--pad=":
            pad_img = int(sys.argv[i][6:])
        elif sys.argv[i][:10] == "--hashpad=":
            hash_pad = int(sys.argv[i][10:])
        elif sys.argv[i][:8] == "--chunk=":
            h5chunksize = int(sys.argv[i][8:])
        elif sys.argv[i][:9] == "--npseed=":
            npseed = int(sys.argv[i][:9])
            np.random.seed(npseed)
        elif sys.argv[i][:4] == "--q=":
            verbosity = int(sys.argv[i][4:])

### TODO -- create a log of parameters used
# txt_out = open("../ " ".txt", "w+")
# log_string = ["* DATASET CREATED WITH THE FOLLOWING PARAMETERS"]


print("numpy random seed set to", npseed)
if not init_ok:
    sys.exit("missing or incorrect command line arguments")
imgsize = y_res // y_pix


def qprint(instr, power):
    if power <= verbosity:
        print(instr)


qprint("running in " + str(imgsize + pad_img * 2) + " * " + str(imgsize + pad_img * 2) + " mode (padding set to " + str(
    pad_img) + ")", 2)

qprint("importing packages", 2)
import os
import math
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import pandas as pd
#from shapely.geometry import mapping
#import rioxarray as rxr
#import xarray as xr
# import geopandas as gpd
# import earthpy as et
# import earthpy.plot as ep
from osgeo import gdal
from osgeo import ogr
import shapefile
import h5py
# from longsgis import voronoiDiagram4plg

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
    arrReshaped = test_img_.reshape(test_img_.shape[0], -1)  # see bookmarked page on how to invert this
    np.savetxt(fs_loc + "/datasrc/x_img/x_test.csv", arrReshaped, delimiter=",", newline="\n")
    print("shape:", arrReshaped.shape)

### generate log of parameters used
txt_param_out = open("../ " ".txt", "w+")
log_param_string = ["* DATASET CREATED WITH THE FOLLOWING PARAMETERS"]
log_param_string += ["  - x_raster_locs () " + str(x_raster_locs)]
log_param_string += ["  - point_shape " + str(point_shape)]
log_param_string += ["  - y_raster_locs " + str(y_raster_loc)]
log_param_string += ["  - y_res " + str(y_res)]
log_param_string += ["  - y_pix " + str(y_pix)]
log_param_string += ["  - create_fs " + str(create_fs)]
log_param_string += ["  - fs_loc " + str(fs_loc)]
log_param_string += ["  - lo_mem " + str(lo_mem)]
log_param_string += ["  - gen_coords " + str(gen_coords)]
log_param_string += ["  - gen_etc " + str(gen_etc)]
log_param_string += ["  - override_regen " + str(override_regen)]
log_param_string += ["  - critical_fields " + str(critical_fields)]
log_param_string += ["  - k_approx " + str(k_approx)]
log_param_string += ["  - testmode " + str(testmode)]
log_param_string += ["  - channel_first " + str(channel_first)]
log_param_string += ["  - pad_img " + str(pad_img)]
log_param_string += ["  - hash_pad " + str(hash_pad)]
log_param_string += ["  - skip_save " + str(skip_save)]
log_param_string += ["  - h5_mode " + str(h5_mode)]
log_param_string += ["  - h5_chunksize " + str(h5chunksize)]
log_param_string += ["  - shuffleorder " + str(shuffleorder)]
log_param_string += ["  - prescreen2 " + str(prescreen2)]
log_param_string += ["  - npseed " + str(npseed)]

"""import sys
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
npseed = None"""

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
yulh, ypxh, _, yulv, _, ypxv = yraster.GetGeoTransform()
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

def cgetter(index, xy):
    if lo_mem:
        return npcoords[index, xy]
    else:
        return grecs[0][index].record[3 - xy]


def clen():
    if lo_mem:
        return npcoords.shape[0]
    else:
        return len(grecs[0])


def pgetter(layer, index):
    # print(layer)
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

# turn coordinates into an index in the data array
def coords_idx(cx, cy, ulh, ulv, psh, psv):
    ix = int((cx - ulh) / psh)
    iy = int((ulv - cy) / psv)
    return ix, iy


# get coordinates of pixel from index
# if mode is 'ctr': get coords of center of pixel
# else if mode is 'ul': get coords of upper left of pixel
def idx_pixctr(ix, iy, ulh, ulv, psh, psv, mode='ul'):
    offsetx = 0
    offsety = 0
    if mode == 'ctr':
        offsetx = psh / 2
        offsety = psv / 2
    cx = ulh + (ix * psh) + offsetx
    cy = ulv - (iy * psv) + offsety
    return cx, cy


def cdist(x1, y1, x2, y2):
    return (((x1 - x2) ** 2) + ((y1 - y2) ** 2))


### stupid k nearest
def getkclose(shapes, centerx, centery, k, ulh, ulv, psh, psv):
    distlist = []
    ids = []
    cx, cy = idx_pixctr(0.5 + centerx, 0.5 + centery, ulh, ulv, psh, psv, mode='ul')
    for i in range(len(shapes)):
        a = shapes[i, 0]
        b = shapes[i, 1]
        distlist.append(cdist(a, b, cx, cy))
        ids.append(i)
    # sort ids by distlist
    ids = [id for _, id in sorted(zip(distlist, ids), key=lambda pair: pair[0])]
    return ids[:k]


maxringsize = 0
avgringsize = 0

### hash all of the gedi footprints! (this could take a while lmao -- rip ram
print("doing the hash thing")
### create it with a buffer

### TODO --- stop at sqrt(2) after the first one
### THE ROOT2 VERSION !!!
mem_root2 = math.sqrt(2)


def krings(x_in, y_in, min_k):
    ring_size = 0
    found_list = []
    cap = -1
    while (cap < 0 or ring_size <= cap):
        i_boundaries = [max(0 - hash_pad, x_in - ring_size), min(yrsize[0] + hash_pad, x_in + ring_size + 1)]
        j_boundaries = [max(0 - hash_pad, y_in - ring_size), min(yrsize[1] + hash_pad, y_in + ring_size + 1)]
        for i in range(i_boundaries[0], i_boundaries[1]):
            for j in range(j_boundaries[0], j_boundaries[1]):
                if i == i_boundaries[0] or i + 1 == i_boundaries[1] or j == j_boundaries[0] or j + 1 == j_boundaries[1]:
                    if len(ygrid_pt_hash[i + 1, j + 1]) > 0:
                        if cap == -1:
                            cap = max(math.ceil(mem_root2 * ring_size), 1) + 1
                        for k in ygrid_pt_hash[i + 1, j + 1]:
                            found_list.append(k)
        ring_size += 1
    return found_list, ring_size


def depricated_krings(x_in, y_in, min_k):
    ring_size = 0
    found_list = []
    while (len(found_list) < min_k):
        i_boundaries = [max(-1, x_in - ring_size), min(yrsize[0] + 1, x_in + ring_size + 1)]
        j_boundaries = [max(-1, y_in - ring_size), min(yrsize[1] + 1, y_in + ring_size + 1)]
        # print(i_boundaries, j_boundaries, x_in, y_in)
        for i in range(i_boundaries[0], i_boundaries[1]):
            for j in range(j_boundaries[0], j_boundaries[1]):
                if i == i_boundaries[0] or i + 1 == i_boundaries[1] or j == j_boundaries[0] or j + 1 == j_boundaries[1]:
                    for k in ygrid_pt_hash[i + 1, j + 1]:
                        # if i in i_boundaries or j in j_boundaries or i+1 in i_boundaries or j+1 in j_boundaries:
                        found_list.append(k)
                        # print("got one")
        # print(ring_size)
        ring_size += 1
    return found_list, ring_size


### ok now actually build the data
# if testmode > 0:
#    pr_unit = testmode // 50
# else:
### TODO --- make this 16x16 instead of 14x14... add 1 unit of buffer on each side
pr_unit = (yrsize[0] * yrsize[1]) // 50
qprint("each step represents " + str(pr_unit) + " samples generated", 1)
progress = 0
nsuccess = 0
extreme_encounter = [0 for ii in range(len(xr_npar) + len(ptlayers) + 2)]
database = []
channels = len(xr_npar) + len(ptlayers) + 2
pd_colnames = ["filename", "y_value", "file_index", "yraster_x", "yraster_y", "avg_mid_dist"]
landmark_x, landmark_y = coords_idx(-104.876653, 41.139535, yulh, yulv, ypxh, ypxv)
if skip_save:
    print("warning: running in skip save mode")

if h5_mode:
    # h5chunksize=1000
    print("running in h5 mode!")
    os.system("rm " + fs_loc + "/datasrc/x_h5.h5")
    h5_dataset = h5py.File(fs_loc + "/datasrc/x_h5.h5", "a")
    if channel_first:
        h5dset = h5_dataset.create_dataset("data",
                                           (h5chunksize, channels, imgsize + (2 * pad_img), imgsize + (2 * pad_img)),
                                           maxshape=(None, channels, imgsize + (2 * pad_img), imgsize + (2 * pad_img)),
                                           chunks=(
                                           h5chunksize, channels, imgsize + (2 * pad_img), imgsize + (2 * pad_img)))
    else:
        h5dset = h5_dataset.create_dataset("data",
                                           (h5chunksize, imgsize + (2 * pad_img), imgsize + (2 * pad_img), channels),
                                           maxshape=(None, imgsize + (2 * pad_img), imgsize + (2 * pad_img), channels),
                                           chunks=(
                                           h5chunksize, imgsize + (2 * pad_img), imgsize + (2 * pad_img), channels))
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

# if verbosity > 0:
#    print("progress 0/50 ", end="", flush=True)

prescreen1 = False
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

### INTERPOLATION PARAMS
xsq_res = 30
ysq_res = 70
xsq_num = 4

#python mcs_interpolation.py ../data/raster/srtm_clipped_co.tif,../data/raster/nlcd_clipped_co.tif ../data/point/GEDI_2B_clean.shp ../data/raster/WUE_Median_Composite_AOI.tif 70 5 true ../data/data_interpolated --lomem --gencoords --override --genetc -c cover,pavd,fhd -q 2 -t 10

print("testing on ... samples", testmode)


#iterate over potentially shuffled x, y
for i in irange_default:
    for j in jrange_default:

        progress += 1
        temp_lcpercent = 0
        extreme_warning = False
        #if the value at the current y is not the nodata value
        if y_npar[i, j] != yndv:
            """if not h5_mode or (h5_mode and h5_scsv):
                if channel_first:
                    x_img = np.zeros((channels, imgsize + (2 * pad_img), imgsize + (2 * pad_img)))
                else:
                    x_img = np.zeros((imgsize + (2 * pad_img), imgsize + (2 * pad_img), channels))
            nlcd_count = 0
            """


            ### Assume csv
            m1_ximg = np.zeros((xsq_num, xsq_num, len(xr_npar)))
            m2_ximg = np.zeros((xsq_num, xsq_num, len(xr_npar)))
            og_ximg = np.zeros((imgsize + (2 * pad_img), imgsize + (2 * pad_img), channels))
            ### we have i, j for y...
            ### build 4x4 grid
            sq_relative = []
            sq_start = [ysq_res/2 - ((xsq_num/2) * xsq_res), ysq_res/2 - ((xsq_num/2) * xsq_res)] # eg 35 - 2*30 = -25/70
            for ii in range(xsq_num):
                sq_relative.append([])
                for jj in range(xsq_num):
                    sq_relative[ii].append([(sq_start[0] + ((ii + 0.5) * xsq_res))/ysq_res,
                                            (sq_start[1] + ((jj + 0.5) * xsq_res))/ysq_res])
            ### so we should have (-10, -10) (20, -10) (etc..)
            ### at each of these points we need the 4 closest in actual crs...?
            ### METHOD 1 - CONVEX COMBO
            ### METHOD 2 - BASIC SAMPLING
            for ii in range(len(sq_relative)):
                for jj in range(len(sq_relative[ii])):
                    ### convert fractional index corresponding to centerpoints of ideal raster grid to crs
                    tempx, tempy = idx_pixctr(sq_relative[ii][jj][0], sq_relative[ii][jj][1], yulh, yulv, ypxh, ypxv, mode='ul')
                    for k in range(len(xr_npar)):
                        ### convert centerpoint coords in crs to index in raster layer (upper left)
                        tempi, tempj = coords_idx(tempx, tempy, xr_params[k][0], xr_params[k][1],
                                                  xr_params[k][2], xr_params[k][3])
                        ### get centerpoint of raster pixel in crs. Location relative to tempx, tempy will help ...
                        ### ... to determine other 3 closest centerpoints
                        ### stands for rasterxcenter, etc
                        rstxc, rstyc = idx_pixctr(tempi, tempj, xr_params[k][0], xr_params[k][1],
                                                  xr_params[k][2], xr_params[k][3], mode='ctr')

                        ### get ids for convex combo
                        ### ideal center further left than box it falls in... need to go 1 index left (<)
                        sqxdiff = tempx - rstxc
                        sqydiff = tempy - rstyc
                        if sqxdiff <= 0: #tx - rx < 0 => tx < rx...
                            sq_refi = -1
                        else:
                            sq_refi = 1
                        ### ideal center further up than box it falls in... need to go 1 index up (<)
                        if sqydiff <= 0:
                            sq_refj = -1
                        else:
                            sq_refj = 1
                        sqwidthx = xr_params[k][2]  # x raster pixel width (horizontal)
                        sqwidthy = xr_params[k][3]  # x raster pixel height (vertical)
                        sq_refs = [(tempi, tempj), (tempi+sq_refi, tempj), (tempi, tempj+sq_refj),
                                   (tempi+sq_refi, tempj+sq_refi)]
                        sqweights = [sqxdiff**2 + sqydiff**2, (sqwidthx-sqxdiff)**2 + sqydiff**2,
                                      sqxdiff**2 + (sqwidthy - sqydiff)**2, (sqwidthx-sqxdiff)**2 + (sqwidthy - sqydiff)**2]
                        sqnorm = sum(sqweights)
                        value = 0
                        print(len(xr_npar))
                        print(k)
                        for sqw in range(4):
                            value += (sqweights[sqw] / sqnorm) * (xr_npar[k][sq_refs[sqw][0], sq_refs[sqw][1]])

                        m1_ximg[ii, jj, k] = value
                        m2_ximg[ii, jj, k] = xr_npar[k][tempi, tempj]

            ### OG METHOD - 5m SAMPLING
            for k in range(len(xr_npar)):
                ### ... Try again with a buffer to get 16x16 image
                for si in range(0 - pad_img, imgsize + pad_img):
                    for sj in range(0 - pad_img, imgsize + pad_img):
                        ### want -.5, .5, 1.5, 2.5, etc...
                        sxoffset = ((2 * si) + 1) / (2 * imgsize)
                        syoffset = ((2 * sj) + 1) / (2 * imgsize)
                        tempx, tempy = idx_pixctr(i + sxoffset, j + syoffset, yulh, yulv, ypxh,
                                                  ypxv, mode='ul')
                        tempi, tempj = coords_idx(tempx, tempy, xr_params[k][0], xr_params[k][1],
                                                  xr_params[k][2], xr_params[k][3])

                        og_ximg[si + pad_img, sj + pad_img, k] = xr_npar[k][tempi, tempj]

            ### make a string of
            ### file name, y value, nsuccess, y raster coordinates, ..., average distance to nearest neighbor
            # good to save
            if not skip_save:
                database.append(["/datasrc/x_img/x_" + str(nsuccess) + ".csv", y_npar[i, j], nsuccess, i, j])
                np.savetxt(fs_loc + "/datasrc/m1_ximg/x_" + str(nsuccess) + ".csv",
                           m1_ximg.reshape(-1, m1_ximg.shape[2]),
                           delimiter=",", newline="\n")
                np.savetxt(fs_loc + "/datasrc/m2_ximg/x_" + str(nsuccess) + ".csv",
                           m2_ximg.reshape(-1, m2_ximg.shape[2]),
                           delimiter=",", newline="\n")
                np.savetxt(fs_loc + "/datasrc/og_ximg/x_" + str(nsuccess) + ".csv",
                           og_ximg.reshape(-1, og_ximg.shape[2]),
                           delimiter=",", newline="\n")
            nsuccess += 1
            if verbosity > 0 and nsuccess % (testmode // 50) == 0:
                print("-", end="", flush=True)
                # else:
                #    dbins = list(failsafe_copy)
            if testmode > 0 and nsuccess > testmode:
                print()
                print("max ring size: ", maxringsize)
                print("avg ring size: ", avgringsize // nsuccess)
                print("saving ydata")
                ydataframe = pd.DataFrame(data=database, columns=pd_colnames)
                ydataframe.to_csv(fs_loc + "/datasrc/ydata.csv")
                # save h5set
                # print("max ring size: ", maxringsize)
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
                # rasters_names_list = ["srtm", "nlcd", "slope", "aspect", "ecostress_WUE"]
                # for idb in range(len(dbins)):
                #    plt.figure()
                #    plt.bar(dbins[idb])
                #    plt.title(rasters_names_list[idb] + " distribution over 5m pixels")
                #    plt.savefig("../figures/pixel_distributions/" + rasters_names_list[idb] + "_dbn.png")
                #    plt.cla()
                #    plt.close()

                sys.exit("exiting after testmode samples")

print()
# print(maxringsize)
"""
print("max ring size: ", maxringsize)
print("saving ydata")
ydataframe = pd.DataFrame(data=database, columns=pd_colnames)
ydataframe.to_csv(fs_loc + "/datasrc/ydata.csv")"""

print("max ring size: ", maxringsize)
print("avg ring size: ", avgringsize // nsuccess)
print("saving ydata")
ydataframe = pd.DataFrame(data=database, columns=pd_colnames)
ydataframe.to_csv(fs_loc + "/datasrc/ydata.csv")
# save h5set
if h5_mode:
    ###need to make sure last chunk is saved
    print("saving last h5 chunk...")
    h5dset.resize(h5len, axis=0)
    h5dset[h5len - h5tid:h5len, :, :, :] = h5_chunk[:h5tid, :, :, :]
    print("saving h5 dset...")
    h5_dataset.close()
# print("max ring size: ", maxringsize)
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