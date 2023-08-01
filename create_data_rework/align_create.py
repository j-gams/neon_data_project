### Written by Jerry Gammie @j-gams
import multiprocessing
import sys
import numpy as np
import align_create_helpers as ach

### TODO -- incorporate meta file


def isf(s):
    if s == "F" or s == "f" or s == "False" or s == "false":
        return True
    return False


def ist(s):
    if s == "T" or s == "t" or s == "True" or s == "true":
        return True
    return False

stopstep = -1

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
if len(sys.argv) < 0:#8:
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

### TODO -- deal with np random seeds
print("numpy random seed set to", npseed)
if not init_ok:
    sys.exit("missing or incorrect command line arguments")
imgsize = y_res // y_pix


def qprint(instr, power):
    if power <= verbosity:
        print(instr)


qprint("running in " + str(imgsize + pad_img * 2) + " * " + str(imgsize + pad_img * 2) + " mode (padding set to " + str(
    pad_img) + ")", 2)

print("importing packages...")
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import shapefile
import h5py
from multiprocessing import Pool

from rfdata_loader import rfloader, piloader

gdal.UseExceptions()

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

if stopstep == 0:
    print("breaking at point 0")
    sys.exit(0)

### load x raster
xraster = []
xr_crs = []
layernames = []
ndv_vals = []
xr_rsize = []
xr_params = []
xr_npar = []

qprint("loading raster data", 1)
for loc in x_raster_locs:
    tdataname = loc.split("/")[-1]
    qprint("loading " + tdataname + " data...", 2)
    print(loc)
    xraster.append(gdal.Open(loc))
    layernames.append(tdataname.split(".")[0])
    tdata_rband = xraster[-1].GetRasterBand(1)
    ndv_vals.append(tdata_rband.GetNoDataValue())
    xr_rsize.append((xraster[-1].RasterXSize, xraster[-1].RasterYSize))
    tulh, tpxh, _, tulv, _, tpxv = xraster[-1].GetGeoTransform()
    tpxv = abs(tpxv)
    ###print crs info
    xr_crs.append(xraster[-1].GetProjection())
    xr_params.append((tulh, tulv, tpxh, tpxv))
    xr_npar.append(xraster[-1].ReadAsArray().transpose())

if stopstep == 1:
    print("breaking at point 1")
    sys.exit(0)

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
yr_crs = yraster.GetProjection()
print(yulh, yulv, ypxh, ypxv)
y_npar = yraster.ReadAsArray().transpose()

if stopstep == 2:
    print("breaking at point 2")
    sys.exit(0)

### TODO -- for multiple pointfiles, need to adjust to point_layers[i][...]
x_point_files = []
point_layers = []
point_layer_indexer = {}
point_records = []
point_layer_names = []

### new variable names
### gen_coords -> reformat coords
### gen_etc -> reformat_data
### grecs -> point records
reformat_data = True
reformat_coords = True
low_memory = True
file_structure_loc = fs_loc
multiprocess = True

### REWORK
# - Iterate over all pointfiles
#   - Index relevant fields in pointfile (and save)
#
"""
for pt in point_shape:
    ### INDEX POINTFILE FIELDS
    if not lo_mem or (reformat_data or reformat_coords):

        temp_dataname = pt.split("/")[-1]
        x_point_files.append(shapefile.Reader(pt))
        point_records.append(x_point_files[-1].shapeRecords())
        ### farm point indexer out to align_create_helpers.py
        ptls, ptlns, ptli, = ach.make_pt_indexer(critical_fields, x_point_files[-1],
                                                 len(x_point_files), file_structure_loc, "")
        point_layers.append(ptls)
        point_layer_names.append(ptlns)
        point_layer_indexer.update(ptli)

    ### Generate reformatted coordinate file
    if low_memory:
        if reformat_coords and not found_coord:
            temp_dispatch = [(0, point_records[-1])]
            temp_dispatch += [(i, x_point_files[-1], point_records[-1], critical_fields[i-1])
                              for i in range(1, len(critical_fields) + 1)]
            if multiprocess and __name__ == '__main__':
                with Pool(multiprocessing.cpu_count()) as ptp:
                    res = ptp.map(ach.reformat_helper, temp_dispatch)
                    ptp.close()
            else:
                res = []
                for elt in temp_dispatch:
                    res.append(ach.reformat_helper(elt))
            ### deal with results
            geo_coord_file = open(fs_loc + "/point_reformat/geo_coords.txt", "w")
            geo_coord_file.write(res[0])
            geo_coord_file.close()
            for i in range(1, len(res)):
                point_data_file = open(fs_loc + "/point_reformat/pt_" + critical_fields[i-1] + ".txt", "w")
                point_data_file.write(res[i])
                point_data_file.close()
        else:
            pass
        if not low_memory or (reformat_coords or reformat_data):
            del x_point_files[-1]
            del point_records[-1]

if stopstep == 3:
    print("breaking at point 3")
    sys.exit(0)

### SAVE CHANNEL NAMES

ptlayers = []
### now try to load the data back in
if lo_mem:
    ### clear hi-mem data (moved to above)
    print("still in lo-memory mode")
    npcoords, _ = rfloader(fs_loc + '/point_reformat/geo_coords.txt')
    # npcoords = np.genfromtxt(fs_loc + '/point_reformat/geo_coords.txt', delimiter=',')
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
        # crit_npar.append(rfloader(fs_loc + '/point_reformat/pt_' + critical_fields[i] + '.txt'))
        crit_npar.append(loaddata)
        ptlayers += fnames
        del loaddata

        # crit_npar.append(np.genfromtxt(fs_loc + '/point_reformat/pt_' + critical_fields[i] + '.txt', delimiter=','))
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
"""
if stopstep == 4:
    print("breaking at point 4")
    sys.exit(0)

maxringsize = 0
avgringsize = 0

def cgetter(index, xy):
    if lo_mem:
        return npcoords[index, xy]
    else:
        return point_records[0][index].record[3 - xy]

def pgetter(layer, index):
    # print(layer)
    if lo_mem:
        ### use crit_npar
        return crit_npar[ptindexer[layer][1]][index, ptindexer[layer][3]]
    else:
        return point_records[ptindexer[layer][0]][index].record[ptindexer[layer][2]]

def clen():
    if lo_mem:
        return npcoords.shape[0]
    else:
        return len(point_records[0])

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
"""
### hash all of the gedi footprints! (this could take a while lmao -- rip ram)
print("doing the hash thing")
### create it with a buffer
ygrid_pt_hash = np.zeros((yrsize[0] + (2 * hash_pad), yrsize[1] + (2 * hash_pad)), dtype='object')
for i in range(ygrid_pt_hash.shape[0]):
    for j in range(ygrid_pt_hash.shape[1]):
        ygrid_pt_hash[i, j] = []
# ygrid_pt_hash = [list([list([]) for ii in range(yrsize[1] + 2)]) for jj in range(yrsize[0] + 2)]
# ygrid_pt_hash[0][0].append(5)
pstep = clen() // 50
print(clen())
actual_added = 0
for i in range(clen()):
    ### get coordinates
    if i % pstep == 0:
        print("-", end="", flush=True)
    xi, yi = ach.coords_idx(cgetter(i, 0), cgetter(i, 1), yulh, yulv, ypxh, ypxv)
    if xi + hash_pad < 0 or yi + hash_pad < 0 or xi > yrsize[0] + (hash_pad * 2) or yi > yrsize[1] + (2 * hash_pad):
        print("big uh-oh!!!")
        print(i)
        print(cgetter(i, 0), cgetter(i, 1))
        print(xi, yi)
    else:
        actual_added += 1
        ygrid_pt_hash[xi + hash_pad, yi + hash_pad].append(i)
print("done doing the hash thing")
print("actually added", actual_added, "gedi points to hash (within bounds)")
"""
### TODO --- stop at sqrt(2) after the first one
### THE ROOT2 VERSION !!!
mem_root2 = math.sqrt(2)

if stopstep == 5:
    print("breaking at point 5")
    sys.exit(0)

###


### TODO --- make this 16x16 instead of 14x14... add 1 unit of buffer on each side
pr_unit = (yrsize[0] * yrsize[1]) // 50
qprint("each step represents " + str(pr_unit) + " samples generated", 1)
progress = 0
nsuccess = 0
#extreme_encounter = [0 for ii in range(len(xr_npar) + len(ptlayers) + 2)]
database = []
#channels = len(xr_npar) + len(ptlayers) + 2
pd_colnames = ["filename", "y_value", "file_index", "yraster_x", "yraster_y", "avg_mid_dist"]
landmark_x, landmark_y = ach.coords_idx(-104.876653, 41.139535, yulh, yulv, ypxh, ypxv)
if skip_save:
    print("warning: running in skip save mode")

"""
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
"""
diids = [ii for ii in range(101)]
dists = [0 for ii in range(101)]

dbins = [[0 for jj in range(20)] for ii in range(len(xr_npar))]
dbinsmins = [0, 0, 0, 0, 0]
dbinsmaxs = [4500, 260, 70, 360, 3.5]

if stopstep == 6:
    print("breaking at point 6")
    sys.exit(0)

###
# what do we need here:
# - relevant layer
# - sampling method
# - grid density
# - name
# - params xr_params.append((tulh, tulv, tpxh, tpxv))

y_crs_pack = [(yulh, yulv, ypxh, ypxv), yr_crs]
if False:
    x_crs_pack = [[xr_params[0], xr_crs[0]],
                  [xr_params[1], xr_crs[1]],
                  [xr_params[2], xr_crs[2]],
                  [xr_params[3], xr_crs[3]]]
else:
    x_crs_pack = [[xr_params[0], xr_crs[0]]]

lads = [[0, ach.alignment_sampling, 30, "SRTM_30_globalUL", ("ul")]]
        #[2, ach.alignment_sampling, 30, "Slope_30_globalUL", ("ul")],
        #[3, ach.alignment_sampling, 30, "Aspect_30_globalUL", ("ul")],
        #[1, ach.alignment_sampling, 30, "NLCD_30_globalUL", ("ul")]]
        #[0, ach.alignment_sampling, 10, "SRTM_10_globalUL", ("ul")],
        #[1, ach.alignment_sampling, 10, "Slope_10_globalUL", ("ul")],
        #[2, ach.alignment_sampling, 10, "Aspect_10_globalUL", ("ul")],
        #[3, ach.alignment_sampling, 10, "NLCD_10_globalUL", ("ul")]]

def save_raster(path, band_count, bands, srs, gt, format='GTiff', dtype = gdal.GDT_Float32):
    cols,rows = bands.shape
    # Initialize driver & create file
    driver = gdal.GetDriverByName(format)
    dataset_out = driver.Create(path, cols, rows, 1, dtype)
    dataset_out.SetGeoTransform(gt)
    dataset_out.SetProjection(srs)
    # Write file to disk
    dataset_out.GetRasterBand(1).WriteArray(bands)
    dataset_out = None

collection_name = "geotifs_sampled_2"
prefix = "../data/" + collection_name
os.system("mkdir " + prefix)

driver = gdal.GetDriverByName("GTiff")
for i in range(len(lads)):
    ### initialize aligner
    layer_geotif = lads[i][1](lads[i][2], (yrsize[0], yrsize[1]), 70, y_crs_pack,
                              x_crs_pack[lads[i][0]], lads[i][4])
    ### perform alignment
    layer_geotif.align(xr_npar[lads[i][0]], subset=-1)

    ### obtain layer geotifs
    driver = gdal.GetDriverByName("GTiff")
    outname = prefix + "/" + lads[i][3] + ".tif"
    layer_out = driver.Create(outname, layer_geotif.data.shape[0], layer_geotif.data.shape[1], 1, gdal.GDT_Float32)
    layer_out.SetGeoTransform(layer_geotif.newcrs)
    layer_out.SetProjection(layer_geotif.newproj)
    ### save layer geotif
    layer_out.GetRasterBand(1).WriteArray(layer_geotif.data.transpose())
    layer_out.FlushCache()

    print("saved to geotif" + outname)
