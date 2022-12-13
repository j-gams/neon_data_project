### Written by Jerry Gammie @j-gams

### DESC...
###

def isf(s):
    if s == "F" or s == "f" or s == "False" or s == "false":
        return True
    return False

def ist(s):
    if s == "T" or s == "t" or s == "True" or s == "true":
        return True
    return False

def qprint(instr, power):
    if power <= verbosity:
        print(instr)

init_ok = True

### Core parameters
x_raster_locs = []
point_shape = []
y_raster_loc = ""
y_res = 70
y_pix = 5
create_fs = True
fs_loc = "../data/default"

### T/F parameters
lo_mem = True
gen_coords = False
gen_etc = False
override_regen = False
skip_save = False
shuffleorder = True
prescreen2 = False
parallelize = False

### Parameters
h5_mode = False
h5_scsv = False
fields_dsets = None
critical_fields = []
testmode = -1
channel_first = False
pad_img = 0
hash_pad = 1
h5chunksize = 1000
npseed = None
npseed_set = False
import_root = None
minimode = False
verbosity = 2

import sys

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
        ### T/F arguments
        ### run in low memory mode
        if sys.argv[i] == "--lomem":
            lo_mem = True
        ### generate list of coordinates (helps with memory, speeds up the process the second time)
        elif sys.argv[i] == "--gencoords":
            gen_coords = True
        ### generate list of other things (see above)
        elif sys.argv[i] == "--genetc":
            gen_etc = True
        ### override existing pre-existing restructured data files
        elif sys.argv[i] == "--override":
            override_regen = True
        ### run without saving (usefule for testing without using a lot of storage)
        elif sys.argv[i] == "--skipsave":
            skip_save = True
        ### run without shuffling data order
        elif sys.argv[i] == "--noshuffle":
            shuffleorder = False
        ### prescreen data (don't save if extreme values are encountered
        elif sys.argv[i] == "--prescreen":
            prescreen2 = True
        ### enable multi-processing
        elif sys.argv[i] == "--parallel":
            parallelize = True
        ### set mini mode
        elif sys.argv[i] == "--mini":
            minimode = True

        ### Other arguments
        ### run in h5 mode / save csv in addition to h5
        elif sys.argv[i][:9] == "--h5mode=":
            if sys.argv[i][9:] == "h5":
                h5_mode = True
                h5_scsv = False
            elif sys.argv[i][9:] == "both":
                h5_mode = True
                h5_scsv = True
        ### comma-separated list of critical fields
        elif sys.argv[i][:10] == "--cfields=":
            ### TODO -- more nuane here to allow multiple pointfiles
            fields_dsets = sys.argv[i][10:].split(";")
            for elt in fields_dsets:
                critical_fields.append(elt.split(","))
        ### run in test mode (generate a subset of possible samples
        elif sys.argv[i][:7] == "--test=":
            testmode = int(sys.argv[i][7:])
        ### orientation - generate images in chw or hwc format
        elif sys.argv[i][:9] == "--orient=":
            if sys.argv[i][9:] == "chw":
                channel_first = True
            else:
                channel_first = False
        ### pad images with data from surrounding areas
        elif sys.argv[i][:6] == "--pad=":
            pad_img = int(sys.argv[i][6:])
        ### pad hash table
        elif sys.argv[i][:10] == "--hashpad=":
            hash_pad = int(sys.argv[i][10:])
        ### set the chunk size used to build big hdf5 file
        elif sys.argv[i][:8] == "--chunk=":
            h5chunksize = int(sys.argv[i][8:])
        ### set the seed for the numpy random functions
        elif sys.argv[i][:9] == "--npseed=":
            npseed = int(sys.argv[i][9:])
            npseed_set = True
        elif sys.argv[i][:9] == "--imroot=":
            import_root = sys.argv[i][9:]
        ### set the verbosity
        elif sys.argv[i][:4] == "--q=":
            verbosity = int(sys.argv[i][4:])

def qprint(instr, power):
    if power <= verbosity:
        print(instr)

### generate log of parameters used
def make_log_list(source_dir):
    txt_param_out = open(source_dir + "/log.txt", "w+")
    log_param_string =  ["* DATASET CREATED WITH THE FOLLOWING PARAMETERS"]
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
    log_param_string += ["  - skip_save " + str(skip_save)]
    log_param_string += ["  - shuffleorder " + str(shuffleorder)]
    log_param_string += ["  - prescreen2 " + str(prescreen2)]
    log_param_string += ["  - parallelize" + str(parallelize)]
    log_param_string += ["  - h5_mode " + str(h5_mode)]
    log_param_string += ["  - h5_save_csv " + str(h5_scsv)]
    log_param_string += ["  - critical_fields " + str(critical_fields)]
    log_param_string += ["  - testmode " + str(testmode)]
    log_param_string += ["  - channel_first " + str(channel_first)]
    log_param_string += ["  - pad_img " + str(pad_img)]
    log_param_string += ["  - hash_pad " + str(hash_pad)]
    log_param_string += ["  - h5_chunksize " + str(h5chunksize)]
    log_param_string += ["  - npseed " + str(npseed)]
    log_param_string += ["  - import_root: " + str(import_root)]
    for lineout in log_param_string:
        txt_param_out.write(lineout + "\n")
    qprint("saving parameter log to meta/log.txt", 2)

if not init_ok:
    sys.exit("missing or incorrect command line arguments")

imgsize = y_res // y_pix

qprint(print("running in " + str(imgsize + pad_img*2) + " * " + str(imgsize + pad_img*2) +
      " mode (padding set to " + str(pad_img) + ")", 2), 2)
if channel_first:
    qprint("running in channel-first mode (chw)", 2)
else:
    qprint("running in channel-third mode (hwc)", 2)
if npseed_set:
    qprint("using ramdom seed: " + str(npseed), 2)

if minimode:
    imgsize = 4
    pad_img = 0
    qprint("running in mini mode...", 2)

qprint("importing packages", 2)
### requirements
import os
import math
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from osgeo import gdal
import shapefile
import h5py
from joblib import Parallel, delayed

### custom help
from rfdata_loader import rfloader, piloader

qprint("done importing packages", 1)

found_coord = False
found_etc = [[False for i in range(len(critical_fields[j]))] for j in range(len(critical_fields))]

### Prints related to creating the directory structure
if lo_mem:
    if gen_coords:
        qprint("generating restructured coordinate file", 2)
    else:
        qprint("not restructuring coordinates", 2)
    if gen_etc:
        qprint("generating restructured files for critical fields", 2)
    else:
        qprint("not restructuring data fields", 2)

qprint("critical fields: " + str(critical_fields), 2)

### TODO -- import coords, etc from another data dir

### create the directory structure
if create_fs:
    ### if the directory structure isn't there, create the basic directories
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
    ### if we are in low-memory mode ... want to avoid loading big stuff into memory
    if lo_mem:
        ### if we want to reformat / resave the coordinates of point data
        if gen_coords:
            ### if we want to delete existing files and start over regardless
            if override_regen:
                qprint("forcibly creating file for reformatted coordinate data", 2)
                os.system("rm " + fs_loc + "/point_reformat/geo_coords.txt")
                os.system("touch " + fs_loc + "/point_reformat/geo_coords.txt")
            ### if we can't find any existing files
            elif not os.path.exists(fs_loc + "/point_reformat/geo_coords.txt"):
                qprint("no file detected, creating file for reformatted coordinate data", 2)
                os.system("touch " + fs_loc + "/point_reformat/geo_coords.txt")
            ### if we find existing files and want to start from there
            else:
                qprint("reformatted coordinate file detected, skipping.", 2)
                found_coord = True
        ### same as above but for critical field data
        if gen_etc:
            for i in range(len(critical_fields)):
                for j in range(len(critical_fields[i])):
                    ### if we want to wipe and start over
                    if override_regen:
                        qprint("forcibly creating file for reformatted " +
                               critical_fields[i][j] + " data " + str(i), 2)
                        os.system("rm " + fs_loc + "/point_reformat/pt_" + str(i) +
                                  "_" + critical_fields[i][j] + ".txt")
                        os.system("touch " + fs_loc + "/point_reformat/pt_" + str(i) +
                                  "_" + critical_fields[i][j] + ".txt")
                    ### if we can't find existing files
                    elif not os.path.exists(fs_loc + "/point_reformat/pt_" + str(i) +
                                  "_" + critical_fields[i][j] + ".txt"):
                        qprint("no file detected, creating file for reformatted " +
                               critical_fields[i][j] + " data " + str(i), 2)
                        os.system("touch " + fs_loc + "/point_reformat/pt_" + str(i) +
                                  "_" + critical_fields[i][j] + ".txt")
                    ### if we can find and want to use existing files
                    else:
                        qprint("reformatted " + critical_fields[i][j] + " (data " + str(i) +
                               ") file detected, skipping.", 2)
                        found_etc[i][j] = True

    ### create a test image in the directoru to ensure things are working
    test_img_ = np.zeros((3, 14, 14))
    arrReshaped = test_img_.reshape(test_img_.shape[0], -1) #see bookmarked page on how to invert this
    np.savetxt(fs_loc + "/datasrc/x_img/x_test.csv", arrReshaped, delimiter=",", newline="\n")
    qprint("shape:" + str(arrReshaped.shape), 2)

### import files from elsewhere
old_toggle = True
if import_root != None:
    qprint("importing coordinate and field data", 2)
    for i in range(len(critical_fields)):
        for j in range(len(critical_fields[i])):
            if old_toggle:
                if os.path.exists(import_root + "/point_reformat/pt_" + critical_fields[i][j] + ".txt"):
                    os.system("cp " + import_root + "/point_reformat/pt_" +
                              critical_fields[i][j] + ".txt " + fs_loc + "/point_reformat/pt_" +
                              str(i) + "_" + critical_fields[i][j] + ".txt")
                    qprint("copied file " + "pt_" + critical_fields[i][j] +
                           ".txt from " + import_root, 2)
                else:
                    qprint("no such file: " + import_root + "/point_reformat/pt_" + critical_fields[i][j] + ".txt", 2)
            else:
                if os.path.exists(import_root + "/point_reformat/pt_" + str(i) +
                                        "_" + critical_fields[i][j] + ".txt"):

                    os.system("cp " + import_root + "/point_reformat/pt_" + str(i) + "_" +
                                critical_fields[i][j] + ".txt " + fs_loc + "/point_reformat/pt_" +
                                str(i) + "_" + critical_fields[i][j] + ".txt")


                    qprint("copied file " + "pt_" + str(i) + "_" + critical_fields[i][j] +
                        ".txt from " + import_root, 2)
                else:
                    qprint("no such file: " + import_root + "/point_reformat/pt_" + str(i) +
                                    "_" + critical_fields[i][j] + ".txt", 2)


### write a log of the parameters used to create the dataset
make_log_list(fs_loc + "/meta")
qprint("done setting up directory structure", 1)

### raster data setup
xraster = []
layernames = []
ndv_vals = []
xr_rsize = []
xr_params = []
xr_npar = []

### load X raster data
qprint("loading raster data", 2)
for loc in x_raster_locs:
    tdataname = loc.split("/")[-1]
    qprint("loading " + tdataname + " data...", 2)
    xraster.append(gdal.Open(loc))
    layernames.append(tdataname.split(".")[0])
    ### crs info
    tdata_rband = xraster[-1].GetRasterBand(1)
    ndv_vals.append(tdata_rband.GetNoDataValue())
    xr_rsize.append((xraster[-1].RasterXSize, xraster[-1].RasterYSize))
    tulh, tpxh, _, tulv, _, tpxv = xraster[-1].GetGeoTransform()
    tpxv = abs(tpxv)
    xr_params.append((tulh, tulv, tpxh, tpxv))
    xr_npar.append(xraster[-1].ReadAsArray().transpose())

### load y raster data
tyname = y_raster_loc.split("/")[-1]
qprint("loading " + tyname + " data...", 2)
yraster = gdal.Open(y_raster_loc)
yrband = yraster.GetRasterBand(1)
yndv = yrband.GetNoDataValue()
qprint("y no-data value: " + str(yndv), 2)
### crs info
yrsize = (yraster.RasterXSize, yraster.RasterYSize)
qprint("y raster dimensions: " + str(yrsize), 2)
yulh, ypxh, _, yulv, _, ypxv =yraster.GetGeoTransform()
ypxv = abs(ypxv)
### print crs info
qprint("y crs info:" + str(yulh) + ", " + str(yulv) + ", " + str(ypxh) + ", " + str(ypxv), 2)
y_npar = yraster.ReadAsArray().transpose()

qprint ("done loading basic raster data", 2)

### setup for dealing with point data
xpoints = []
ptlayers = []
ptindexer = {}
grecs = []
ptlnames = []

def field_parallel_dispatch(i_disp, gen_etc_disp, found_etc_disp, xpoints_disp, crit_fields_disp,
                            grecs_disp, fs_loc_disp, verbosity_disp):
    if gen_etc_disp and not found_etc_disp[i]:
        ### sweet progress bar
        qprint("reformatting critical field " + crit_fields_disp[j][i_disp] + ". This could take a while...", 1)
        #if verbosity > 0:
        #    print("progress 0/50 ", end="", flush=True)
        progress = 0
        ### build list of field indices, names:
        fields_ids = []
        fields_names = ""
        for k in range(len(xpoints_disp[-1].fields)):
            if crit_fields_disp[j][i_disp] in xpoints_disp[-1].fields[k][0]:
                fields_ids.append(k)
                fields_names += xpoints_disp[-1].fields[k][0] + ","
        ### header
        os.system('echo "' + fields_names[:-1] + '" >> ' + fs_loc_disp + "/point_reformat/pt_" +
                  str(j) + "_" + crit_fields_disp[j][i_disp] + ".txt")
        for jz in range(len(grecs_disp[-1])):
            progress += 1
            if verbosity_disp > 0 and progress % (len(grecs_disp[-1]) // 50) == 0:
                print("-", end="", flush=True)
            dumpstr = ""
            for k in fields_ids:
                # for k in range(len(xpoints[-1].fields)):
                if crit_fields_disp[j][i_disp] in xpoints_disp[-1].fields[k][0]:
                    dumpstr += str(grecs_disp[-1][jz].record[k]) + ","
            os.system('echo "' + dumpstr[:-1] + '" >> ' + fs_loc_disp + '/point_reformat/pt_' +
                      str(j) + "_" + crit_fields_disp[j][i_disp] + '.txt')
        qprint("---> 50/50 (done)", 1)
    qprint("done reformatting " + crit_fields_disp[j][i_disp] + " data", 1)

for j in range(len(point_shape)):
    pt = point_shape[j]
    ### only do this if we actually need to load in the big files
    ### if not we can use the reformatted data that is smaller / faster
    if not lo_mem or (gen_coords or gen_etc):
        ### load point data
        tdataname = pt.split("/")[-1]
        qprint("loading " + tdataname + " data...", 2)
        xpoints.append(shapefile.Reader(pt))
        qprint("loading " + tdataname + " records...", 2)
        grecs.append(xpoints[-1].shapeRecords())

        ### for each field we care about, gather the data
        for i in range(len(critical_fields[j])):
            ptl_idx = 0
            for k in range(len(xpoints[-1].fields)):
                if critical_fields[j][i] in xpoints[-1].fields[k][0]:
                    ### given the layer (key) provides the ptfile #, critical field #, field index within that shape,
                    ### and id within the np array (if exists)
                    ptindexer[len(ptlayers)] = (len(xpoints) -1, i, k, ptl_idx)
                    ptlayers.append(critical_fields[j][i] + "_" + str(ptl_idx))
                    ptlnames.append(xpoints[-1].fields[k][0])
                    ptl_idx += 1

        ### save the data we gathered to make life easier later
        qprint("writing point indexer file", 2)
        if os.path.exists(fs_loc + "/point_reformat/pt_indexer_util.txt"):
            os.system("rm " + fs_loc + "/point_reformat/pt_indexer_util.txt")
        ### idea - use method that is not very memory intensive to write this stuff
        ### pretty sure this is slower than alternatives but hey
        for lkey in ptindexer.keys():
            lkstr = ""
            for elt in ptindexer[lkey]:
                lkstr += str(elt) + ","
            os.system('echo "' + str(lkey) + ':' + lkstr[:-1] + '" >> ' + fs_loc + '/point_reformat/pt_indexer_util.txt')

    ### if we are in low memory mode, we want to load data we have already worked with if possible
    if lo_mem:
        if gen_coords and not found_coord:
            ### sweet progress bar for this
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

        ### do the same thing for pointfile field data
        ### TODO -- introduce parallelization
        if parallelize:
            #def field_parallel_dispatch(gen_etc_disp, found_etc_disp, xpoints_disp, crit_fields_disp,
            #                grecs_disp, fs_loc_disp, verbosity_disp):
            Parallel(n_jobs=-1)(delayed(field_parallel_dispatch)(i, gen_etc, found_etc,
                                                                 xpoints, critical_fields,
                                                                 grecs, fs_loc, verbosity) for i in
                                range(len(critical_fields[j])))
        else:
            for i in range(len(critical_fields[j])):
                if gen_etc and not found_etc[j][i]:
                    ### sweet progress bar
                    qprint("reformatting critical field " + critical_fields[j][i] + ". This could take a while...", 1)
                    if verbosity > 0:
                        print("progress 0/50 ", end="", flush=True)
                    progress = 0
                    ### build list of field indices, names:
                    fields_ids = []
                    fields_names = ""
                    for k in range(len(xpoints[-1].fields)):
                        if critical_fields[j][i] in xpoints[-1].fields[k][0]:
                            fields_ids.append(k)
                            fields_names += xpoints[-1].fields[k][0] + ","
                    ### header
                    os.system('echo "' + fields_names[:-1] + '" >> ' + fs_loc + "/point_reformat/pt_" +
                              str(j) + "_" + critical_fields[j][i] + ".txt")
                    for jz in range(len(grecs[-1])):
                        progress += 1
                        if verbosity > 0 and progress % (len(grecs[-1]) // 50) == 0:
                            print("-", end="", flush=True)
                        dumpstr = ""
                        for k in fields_ids:
                        #for k in range(len(xpoints[-1].fields)):
                            if critical_fields[j][i] in xpoints[-1].fields[k][0]:
                                dumpstr += str(grecs[-1][jz].record[k]) + ","
                        os.system('echo "' + dumpstr[:-1] + '" >> ' + fs_loc + '/point_reformat/pt_' +
                                  str(j) + "_" + critical_fields[j][i] + '.txt')
                    qprint("---> 50/50 (done)", 1)
                qprint("done reformatting " + critical_fields[j][i] + " data", 1)
        ### clean up a bit
        if not lo_mem or (gen_coords or gen_etc):
            del xpoints[-1]
            del grecs[-1]

qprint("layer number audit: " + str(len(ptlayers)), 2)
qprint("done with major raw data reformatting", 1)

ptlayers = []
### if we are in low memory mode we have deleted data from memory, or not yet loaded in the first place
### so, load the data back in, in less bloated form:
if lo_mem:
    ### clear hi-mem data (moved to above)
    qprint("low memory mode: loading coordinate data", 2)
    npcoords, _ = rfloader(fs_loc + '/point_reformat/geo_coords.txt')

    crit_npar = []

    ### load point indexer
    qprint("low memory mode: loading point indexer", 2)
    ptindexer = piloader(fs_loc + "/point_reformat/pt_indexer_util.txt")
    ### load in critical field data
    for i in range(len(critical_fields)):
        crit_npar.append([])
        for j in range(len(critical_fields[i])):
            ##load in
            loaddata, fnames = rfloader(fs_loc + '/point_reformat/pt_' + str(i) + "_" +
                                        critical_fields[i][j] + '.txt')
            crit_npar[i].append(loaddata)
            ptlayers += fnames
            ### clean up
            del loaddata

            qprint("critical field data shape: " + str(crit_npar[i][-1].shape), 2)
            if crit_npar[i][-1].ndim < 2:
                cnpdim = len(crit_npar[i][-1])
                crit_npar[i][-1] = np.reshape(crit_npar[i][-1], (-1, 1))
            qprint("critical field nan values: " + str(np.count_nonzero(np.isnan(crit_npar[i][-1]))), 2)
    qprint("low memory layer number audit: " +  str(len(ptlayers)), 2)
    qprint("reloaded coordinate and point field data", 1)

#qprint(crit_npar, 3)
### now we have all the data loaded in?
### SAVE CHANNEL NAMES
save_channelnames = layernames + ptlayers
os.system("rm " + fs_loc + "/meta/channel_names.txt")
os.system("touch " + fs_loc + "/meta/channel_names.txt")
for cname in save_channelnames:
    os.system('echo "' + cname + '," >> ' + fs_loc + '/meta/channel_names.txt')

### some helpers below are necessary because data is in a different format depending on lo-mem
### helper getting the coordinates of the ith point
def cgetter(index, xy):
    if lo_mem:
        return npcoords[index, xy]
    else:
        return grecs[0][index].record[3-xy]

### helper to get the length of the point file
def clen():
    if lo_mem:
        return npcoords.shape[0]
    else:
        return len(grecs[0])

### helper to get the field value at layer (nth field), value
def pgetter(layer, index, i2=0):
    if lo_mem:
        return crit_npar[i2][ptindexer[layer][1]][index, ptindexer[layer][3]]
    else:
        return grecs[ptindexer[layer][0]][index].record[ptindexer[layer][2]]

### in geotif, raster data stored from upper left pixel coordinates, pixel width, and rotation.
### so we can get value at coordinate (x,y) from looking at pixel at
### (x - UL_x)/size_x
### (UL_y - y)/size_y
### here these are switched because in the geographic crs, increasing coordinates go up and to the right
### however within the matrix they go down and to the right, so the sign must be reversed for y

### helper to turn coordinates into an index in the data array
def coords_idx(cx, cy, ulh, ulv, psh, psv):
    ix = int((cx - ulh)/psh)
    iy = int((ulv - cy)/psv)
    return ix, iy

### helper to get coordinates of pixel from index
### if mode is 'ctr': get coords of center of pixel
### else if mode is 'ul': get coords of upper left of pixel
def idx_pixctr(ix, iy, ulh, ulv, psh, psv, mode='ul'):
    offsetx = 0
    offsety = 0
    if mode=='ctr':
        offsetx = psh/2
        offsety = psv/2
    cx = ulh + (ix * psh) + offsetx
    cy = ulv - (iy * psv) + offsety
    return cx, cy

def coverlap(i, j, yulh, yulv, ypxh, ypxv, xulh, xulv, xpxh, xpxv):
    #print(yulh, yulv, ypxh, ypxv)
    ypxh /= 4
    ypxv /= 4
    yulh = yulh + i*ypxh
    yulv = yulv - j*ypxv

    ah = abs(yulh - xulh)
    av = abs(yulv - xulv)
    bh = min(ypxh - xpxh - ah, 0)
    bv = min(ypxv - xpxv - av, 0)
    sideh = xpxh - abs(bh)
    sidev = xpxv - abs(bv)
    xcover = 1 - (((bh * sidev) + (bv * sideh) + (bh * bv))/abs(xpxh * xpxv))
    return xcover


### helper to get euclidean distance between two coordinates
def cdist(x1, y1, x2, y2):
    return (((x1 - x2) ** 2) + ((y1 - y2) ** 2))

maxringsize = 0
avgringsize = 0
qprint("creating the hash array:", 1)
ygrid_pt_hash = np.zeros((yrsize[0] + (2*hash_pad), yrsize[1] + (2*hash_pad)), dtype='object')
for i in range(ygrid_pt_hash.shape[0]):
    for j in range(ygrid_pt_hash.shape[1]):
        ygrid_pt_hash[i, j] = []

### some work for the progress bar...
pstep = clen() // 50
actual_added = 0
### iterate through point data and map to position in hash array
for i in range(clen()):
    ### sweet progress bar
    if i % pstep == 0: #and verbosity > 0
        print("-", end="", flush=True)
    ### get coordinates
    xi, yi = coords_idx(cgetter(i, 0), cgetter(i, 1), yulh, yulv, ypxh, ypxv)

    ### to error
    if xi+hash_pad < 0 or yi+hash_pad < 0 or xi > yrsize[0]+(hash_pad*2) or yi > yrsize[1]+(2*hash_pad):
        print("err: big uh-oh!!!")
        print("err:", i)
        print("err:", cgetter(i, 0), cgetter(i, 1))
        print("err:", xi, yi)
    else:
        actual_added += 1
        ygrid_pt_hash[xi+hash_pad, yi+hash_pad].append(i)

qprint("---> done", 1)
qprint("actually added " + str(actual_added) + " shape points to hash (within bounds)", 2)

### maybe this will speed things up a little bit?
mem_root2 = math.sqrt(2)

### sqrt2 version - stop looking further for nearest neighbors after a distance proportional
### to sqrt(2) * the distance to the first point we find
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

### now... actually build the dataset
### some progress counters
progress = 0
nsuccess = 0

### keep track of abnormally large / small values with this
extreme_encounter = [0 for ii in range(len(xr_npar) + len(ptlayers) + 2)]

### database for y values, metadata
database = []

### TODO -- all channel math needs work to convert to more general ammount of point data
### compute number of channels
channels = len(xr_npar) + len(ptlayers) + 2

### column names for y database
pd_colnames = ["filename", "y_value", "file_index", "yraster_x", "yraster_y", "avg_mid_dist"]

### TODO -- see if this gets used anywhere
### pick a value to look out for ?
landmark_x, landmark_y = coords_idx(-104.876653,41.139535, yulh, yulv, ypxh, ypxv)

### warn the user about skipping saves
if skip_save:
    qprint("warning: running in skip save mode", 1)

### set up the h5 database for x data
if h5_mode:
    qprint("running in h5 mode!", 1)
    os.system("rm " + fs_loc + "/datasrc/x_h5.h5")
    h5_dataset = h5py.File(fs_loc + "/datasrc/x_h5.h5", "a")
    if not minimode:
        if channel_first:
            h5dset = h5_dataset.create_dataset("data", (h5chunksize, channels, imgsize + (2 * pad_img), imgsize + (2 * pad_img)),
                                               maxshape=(None, channels, imgsize+(2*pad_img), imgsize+(2*pad_img)),
                                               chunks=(h5chunksize,channels, imgsize+(2*pad_img), imgsize+(2*pad_img)))
        else:
            h5dset = h5_dataset.create_dataset("data", (h5chunksize, imgsize+(2*pad_img), imgsize+(2*pad_img), channels),
                                               maxshape=(None,imgsize+(2*pad_img), imgsize+(2*pad_img), channels),
                                               chunks=(h5chunksize,imgsize+(2*pad_img), imgsize+(2*pad_img), channels))
    else:
        h5_cdim = int((4 * 2) + (64//16))
        h5dset = h5_dataset.create_dataset("data", (h5chunksize, imgsize, imgsize, h5_cdim),
                                           maxshape=(None, imgsize, imgsize, h5_cdim),
                                           chunks=(
                                           h5chunksize, imgsize, imgsize, h5_cdim))
    h5len = 0
    h5tid = 0
    h5chunkid = 0
    if not minimode:
        if channel_first:
            h5_chunk = np.zeros((h5chunksize, channels, imgsize + (2 * pad_img), imgsize + (2 * pad_img)))
        else:
            h5_chunk = np.zeros((h5chunksize, imgsize + (2 * pad_img), imgsize + (2 * pad_img), channels))
    else:
        h5_chunk = np.zeros((h5chunksize, imgsize, imgsize, h5_cdim))
    h5_chunk.fill(-1)

if testmode > 0:
    qprint("running in test mode on " + str(testmode) + " samples", 1)

### forget what this is for ... probably not important
diids = [ii for ii in range(101)]
dists = [0 for ii in range(101)]
dbins = [[0 for jj in range(20)] for ii in range(len(xr_npar))]
dbinsmins = [0, 0, 0, 0, 0]
dbinsmaxs = [4500, 260, 70, 360, 3.5]

### TODO -- remove if unused
### parameters for prescreen-1 (not currently in use)
prescreen1=False
extreme_bounds = [-100000, 100000]
prescreen_dist = 35 ** 2
prescreen_forestp = 0.95

### create the shuffled indices to randomize order of dataset building
if shuffleorder:
    irange_default = np.arange(yrsize[0])
    jrange_default = np.arange(yrsize[1])
    np.random.shuffle(irange_default)
    np.random.shuffle(jrange_default)
else:
    irange_default = np.arange(yrsize[0])
    jrange_default = np.arange(yrsize[1])

def raster_p(i):
    #for si in range(0 - pad_img, imgsize + pad_img):
    #    for sj in range(0 - pad_img, imgsize + pad_img):
    pass

### iteratively build the dataset
### iterate over y raster dataset

#if parallelize:
#    Parallel(n_jobs=-1)(delayed(parallel_dispatch)(ids) for ids in
#                        np.array(np.meshgrid(irange_default, jrange_default)).T.reshape(-1, 2))
#else:
for i in irange_default:
    for j in jrange_default:
        progress += 1
        extreme_warning = False

        qprint("tick", 3)
        ### only proceed if the y value is not the no-data value
        if y_npar[i, j] != yndv:
            qprint("not ndv", 3)
            ### initialize array for csv - data
            if not h5_mode or (h5_mode and h5_scsv):
                if channel_first:
                    x_img = np.zeros((channels, imgsize+(2*pad_img), imgsize+(2*pad_img)))
                else:
                    x_img = np.zeros((imgsize+(2*pad_img), imgsize+(2*pad_img), channels))
            nlcd_count = 0

            if minimode:
                ###
                qprint(nsuccess)
                for k in range(len(xr_npar)):
                    for si in range(4):
                        for sj in range(4):
                            sxoffset = ((2 * si) + 1) / (2 * imgsize)
                            syoffset = ((2 * sj) + 1) / (2 * imgsize)
                            tempx, tempy = idx_pixctr(i + sxoffset, j + syoffset, yulh, yulv, ypxh,
                                                      ypxv, mode='ul')
                            tempi, tempj = coords_idx(tempx, tempy, xr_params[k][0], xr_params[k][1],
                                                      xr_params[k][2], xr_params[k][3])
                            h5_chunk[h5tid, si, sj, k*2] = xr_npar[k][tempi, tempj]
                            ###overlap
                            h5_chunk[h5tid, si, sj, k*2 + 1] = coverlap(si, sj, yulh, yulv, ypxh, ypxv, xr_params[k][0],
                                                                        xr_params[k][1], xr_params[k][2], xr_params[k][3])
                k_ids, rings = krings(i, j, 0)
                mindist = 1000000
                minpt = None
                ### brute force find nearest neighbor for this pixel from subset
                tempcx, tempcy = idx_pixctr(0.5, 0.5, yulh, yulv, ypxh, ypxv, mode='ul')
                for pt_idx in k_ids:
                    tdist = cdist(npcoords[pt_idx, 0], npcoords[pt_idx, 1], tempcx, tempcy)
                    if tdist < mindist:
                        mindist = tdist
                        minpt = pt_idx
                tptdata = []
                for m in range(len(ptlayers)):
                    tptdata.append(pgetter(m, minpt))
                tptdata.append(minpt)
                tptdata.append(mindist)
                h5_chunk[h5tid, :, :, 4*2:] = np.array(tptdata).reshape((4, 4, -1))
                avg_mid_dist=0
                if (not extreme_warning and prescreen2) or not prescreen2:
                    ### if we actually want to save this point as a .csv
                    if not skip_save and (not h5_mode or (h5_mode and h5_scsv)):
                        if channel_first:
                            np.savetxt(fs_loc + "/datasrc/x_img/x_" + str(nsuccess) + ".csv",
                                       x_img.reshape(x_img.shape[0], -1),
                                       delimiter=",", newline="\n")
                        else:
                            np.savetxt(fs_loc + "/datasrc/x_img/x_" + str(nsuccess) + ".csv",
                                       x_img.reshape(-1, x_img.shape[2]),
                                       delimiter=",", newline="\n")

                    ### if we want to save this point to the h5 database
                    if not skip_save and h5_mode:
                        h5tid += 1
                        h5len += 1

                        ### progress
                        if h5tid % (h5chunksize // 10) == 0 and verbosity > 0:
                            print("-", end="", flush=True)

                        ### if we have completed an entire chunk, we need to copy over the data being
                        ### temporarily stored in the numpy array to the h5 file
                        if h5tid == h5chunksize:
                            h5chunkid += 1
                            qprint("> resizing h5 (" + str(h5chunkid) + ")", 2)

                            ### resize h5 dataset
                            h5dset.resize(h5len, axis=0)

                            ### copy the data over from numpy array
                            h5dset[h5len - h5chunksize:h5len, :, :, :] = np.array(h5_chunk[:, :, :, :])

                            ### reset the chunk
                            h5_chunk = np.zeros(h5_chunk.shape)
                            h5_chunk.fill(-1)
                            h5tid = 0

                    ### if we actually want to save this datapoint, record y value / metadata
                    if not skip_save:
                        database.append(
                            ["/datasrc/x_img/x_" + str(nsuccess) + ".csv", y_npar[i, j], nsuccess, i, j, avg_mid_dist])
                    # qprint("success", 2)
                    nsuccess += 1

                if testmode > 0 and nsuccess > testmode:
                    qprint("", 1)
                    qprint("build summary: max ring size " + str(maxringsize), 1)
                    qprint("build summary: avg ring size " + str(avgringsize // nsuccess), 1)
                    qprint("saving ydata", 2)
                    ### create a dataframe with the y data
                    ydataframe = pd.DataFrame(data=database, columns=pd_colnames)
                    ### save the ydata
                    ydataframe.to_csv(fs_loc + "/datasrc/ydata.csv")
                    qprint("ydata saved", 1)


                    ### save h5set
                    if h5_mode:
                        ### need to make sure last chunk is copied over before saving the file
                        qprint("saving last h5 chunk...", 2)
                        h5dset.resize(h5len, axis=0)
                        h5dset[h5len - h5tid:h5len, :, :, :] = h5_chunk[:h5tid, :, :, :]
                        qprint("saving h5 dset...", 2)
                        h5_dataset.close()

                continue

            ### iterate through every x raster dataset
            for k in range(len(xr_npar)):
                ### iterate over output dimensions (plus padding)
                if parallelize:
                    continue
                else:
                    for si in range(0 - pad_img, imgsize+pad_img):
                        for sj in range(0 - pad_img, imgsize+pad_img):
                            ### want -.5, .5, 1.5, 2.5, etc...
                            ### deal with crs acrobatics
                            sxoffset = ((2 * si) + 1) / (2 * imgsize)
                            syoffset = ((2 * sj) + 1) / (2 * imgsize)
                            tempx, tempy = idx_pixctr(i + sxoffset, j + syoffset, yulh, yulv, ypxh,
                                                      ypxv, mode='ul')
                            tempi, tempj = coords_idx(tempx, tempy, xr_params[k][0], xr_params[k][1],
                                                      xr_params[k][2], xr_params[k][3])
                            ### check extreme encounters
                            if  xr_npar[k][tempi, tempj] > extreme_bounds[1] or xr_npar[k][tempi, tempj] < extreme_bounds[0]:
                                extreme_warning = True
                                extreme_encounter[k] += 1
                            if k == 1 and (xr_npar[k][tempi, tempj] < 40 or xr_npar[k][tempi, tempj] > 45):
                                nlcd_count += 1
                            ### record sampled values
                            if not h5_mode or (h5_mode and h5_scsv):
                                if channel_first:
                                    x_img[k, si + pad_img, sj + pad_img] = xr_npar[k][tempi, tempj]
                                else:
                                    x_img[si + pad_img, sj + pad_img, k] = xr_npar[k][tempi, tempj]
                            if h5_mode:
                                if channel_first:
                                    h5_chunk[h5tid, k, si + pad_img, sj + pad_img] = xr_npar[k][tempi, tempj]
                                else:
                                    h5_chunk[h5tid, si + pad_img, sj + pad_img, k] = xr_npar[k][tempi, tempj]
            qprint("did raster layers", 3)
            ### get subset of nearest neighbors to this grid square
            k_ids, rings = krings(i, j, 0)
            qprint("did krings", 3)
            avgringsize += rings
            ### record maximum ring size encountered
            if rings > maxringsize:
                maxringsize = rings
            ### iterate through pixels within sampple...
            for si in range(0-pad_img, imgsize+pad_img):
                for sj in range(0-pad_img, imgsize+pad_img):
                    ### crs magic
                    sxoffset = ((2 * si) + 1) / (2 * imgsize)
                    syoffset = ((2 * sj) + 1) / (2 * imgsize)
                    tempx, tempy = idx_pixctr(i + sxoffset, j + syoffset, yulh, yulv, ypxh,
                            ypxv, mode='ul')
                    mindist = 1000000
                    minpt = None
                    ### brute force find nearest neighbor for this pixel from subset
                    for pt_idx in k_ids:
                        tdist = cdist(npcoords[pt_idx, 0], npcoords[pt_idx, 1], tempx, tempy)
                        if tdist < mindist:
                            mindist = tdist
                            minpt = pt_idx

                    ### get the data from each field
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

                    ### record the data
                    if not h5_mode or (h5_mode and h5_scsv):
                        if channel_first:
                            x_img[len(xr_npar) + len(ptlayers), si + pad_img, sj + pad_img] = minpt
                            x_img[len(xr_npar) + len(ptlayers) + 1, si + pad_img, sj + pad_img] = mindist
                        else:
                            x_img[si + pad_img, sj + pad_img, len(xr_npar) + len(ptlayers)] = minpt
                            x_img[si + pad_img, sj + pad_img, len(xr_npar) + len(ptlayers) + 1] = mindist
                    if h5_mode:
                        if channel_first:
                            h5_chunk[h5tid, len(xr_npar) + len(ptlayers), si + pad_img, + sj + pad_img] = minpt
                            h5_chunk[h5tid, len(xr_npar) + len(ptlayers) + 1, si + pad_img, + sj + pad_img] = mindist
                        else:
                            h5_chunk[h5tid, si + pad_img, sj + pad_img, len(xr_npar) + len(ptlayers)] = minpt
                            h5_chunk[h5tid, si + pad_img, sj + pad_img, len(xr_npar) + len(ptlayers) + 1] = mindist

            qprint("did points", 3)
            ### keep track of the distance to nearest point centroid from center of the datacube
            if h5_mode:
                if channel_first:
                    avg_mid_dist = h5_chunk[
                                       h5tid, -1, (imgsize + (pad_img * 2)) // 2, (imgsize + (pad_img * 2)) // 2] / 4
                    avg_mid_dist += h5_chunk[h5tid, -1, (imgsize + (pad_img * 2) - 1) // 2, (
                                imgsize + (pad_img * 2)) // 2] / 4
                    avg_mid_dist += h5_chunk[h5tid, -1, (imgsize + (pad_img * 2)) // 2, (
                                imgsize + (pad_img * 2) - 1) // 2] / 4
                    avg_mid_dist += h5_chunk[h5tid, -1, (imgsize + (pad_img * 2) - 1) // 2, (
                                imgsize + (pad_img * 2) - 1) // 2] / 4
                else:
                    avg_mid_dist = h5_chunk[
                                       h5tid, (imgsize + (pad_img * 2)) // 2, (imgsize + (pad_img * 2)) // 2, -1] / 4
                    avg_mid_dist += h5_chunk[h5tid, (imgsize + (pad_img * 2) - 1) // 2, (
                                imgsize + (pad_img * 2)) // 2, -1] / 4
                    avg_mid_dist += h5_chunk[h5tid, (imgsize + (pad_img * 2)) // 2, (
                                imgsize + (pad_img * 2) - 1) // 2, -1] / 4
                    avg_mid_dist += h5_chunk[h5tid, (imgsize + (pad_img * 2) - 1) // 2, (
                                imgsize + (pad_img * 2) - 1) // 2, -1] / 4
            else:
                if channel_first:
                    avg_mid_dist = x_img[-1, (imgsize + (pad_img * 2)) // 2, (imgsize + (pad_img * 2)) // 2] / 4
                    avg_mid_dist += x_img[-1, (imgsize + (pad_img * 2) - 1) // 2, (imgsize + (pad_img * 2)) // 2] / 4
                    avg_mid_dist += x_img[-1, (imgsize + (pad_img * 2)) // 2, (imgsize + (pad_img * 2) - 1) // 2] / 4
                    avg_mid_dist += x_img[
                                        -1, (imgsize + (pad_img * 2) - 1) // 2, (imgsize + (pad_img * 2) - 1) // 2] / 4
                else:
                    avg_mid_dist = x_img[(imgsize + (pad_img * 2)) // 2, (imgsize + (pad_img * 2)) // 2, -1] / 4
                    avg_mid_dist += x_img[(imgsize + (pad_img * 2) - 1) // 2, (imgsize + (pad_img * 2)) // 2, -1] / 4
                    avg_mid_dist += x_img[(imgsize + (pad_img * 2)) // 2, (imgsize + (pad_img * 2) - 1) // 2, -1] / 4
                    avg_mid_dist += x_img[
                                        (imgsize + (pad_img * 2) - 1) // 2, (imgsize + (pad_img * 2) - 1) // 2, -1] / 4
            qprint("did mid avg", 3)
            qprint("tick " + str(extreme_warning) + " " + str(prescreen2), 3)
            ### record y data, metadata to database
            if (not extreme_warning and prescreen2) or not prescreen2:
                ### if we actually want to save this point as a .csv
                if not skip_save and (not h5_mode or (h5_mode and h5_scsv)):
                    if channel_first:
                        np.savetxt(fs_loc + "/datasrc/x_img/x_" +str(nsuccess)+ ".csv", x_img.reshape(x_img.shape[0], -1),
                                delimiter=",", newline="\n")
                    else:
                        np.savetxt(fs_loc + "/datasrc/x_img/x_" +str(nsuccess)+ ".csv", x_img.reshape(-1, x_img.shape[2]),
                                delimiter=",", newline="\n")

                ### if we want to save this point to the h5 database
                if not skip_save and h5_mode:
                    h5tid += 1
                    h5len += 1

                    ### progress
                    if h5tid % (h5chunksize//10) == 0 and verbosity > 0:
                        print("-", end="", flush=True)

                    ### if we have completed an entire chunk, we need to copy over the data being
                    ### temporarily stored in the numpy array to the h5 file
                    if h5tid == h5chunksize:
                        h5chunkid += 1
                        qprint("> resizing h5 (" + str(h5chunkid) + ")", 2)

                        ### resize h5 dataset
                        h5dset.resize(h5len, axis=0)

                        ### copy the data over from numpy array
                        h5dset[h5len-h5chunksize:h5len,:,:,:] = np.array(h5_chunk[:,:,:,:])

                        ### reset the chunk
                        h5_chunk = np.zeros(h5_chunk.shape)
                        h5_chunk.fill(-1)
                        h5tid = 0

                ### if we actually want to save this datapoint, record y value / metadata
                if not skip_save:
                    database.append(["/datasrc/x_img/x_" + str(nsuccess) + ".csv", y_npar[i, j], nsuccess, i, j, avg_mid_dist])
                #qprint("success", 2)
                nsuccess += 1

            ### if we are stoppping early and the conditions are met, stop!
            if testmode > 0 and nsuccess > testmode:
                qprint("", 1)
                qprint("build summary: max ring size " + str(maxringsize), 1)
                qprint("build summary: avg ring size " + str(avgringsize//nsuccess), 1)
                qprint("saving ydata", 2)
                ### create a dataframe with the y data
                ydataframe = pd.DataFrame(data=database, columns=pd_colnames)
                ### save the ydata
                ydataframe.to_csv(fs_loc + "/datasrc/ydata.csv")
                qprint("ydata saved", 1)

                ### save h5set
                if h5_mode:
                    ### need to make sure last chunk is copied over before saving the file
                    qprint("saving last h5 chunk...", 2)
                    h5dset.resize(h5len, axis=0)
                    h5dset[h5len-h5tid:h5len,:,:,:] = h5_chunk[:h5tid,:,:,:]
                    qprint("saving h5 dset...", 2)
                    h5_dataset.close()

                qprint("extreme encounter report:", 2)
                no_enc = True
                ### do some analysis on extreme value encounters
                for i in range(len(extreme_encounter)):
                    if extreme_encounter[i] > 0:
                        no_enc = False
                        qprint("  " +  str(i) + str(extreme_encounter[i]), 2)
                if no_enc:
                    qprint("no extreme encounters", 2)

                ### do some plots
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

                sys.exit("exiting after testmode samples")

qprint("", 1)
qprint("build summary: max ring size " + str(maxringsize), 1)
qprint("build summary: avg ring size " + str(avgringsize//nsuccess), 1)
qprint("saving ydata", 2)

### save h5 dataset
if h5_mode:
    ### need to make sure last chunk is saved
    qprint("saving last h5 chunk...", 2)
    h5dset.resize(h5len, axis=0)
    h5dset[h5len-h5tid:h5len,:,:,:] = h5_chunk[:h5tid,:,:,:]
    qprint("saving h5 dset...", 2)
    h5_dataset.close()

qprint("extreme encounter report:", 2)
no_enc = True
### do some analysis on extreme value encounters
for i in range(len(extreme_encounter)):
    if extreme_encounter[i] > 0:
        no_enc = False
        qprint("  " +  str(i) + str(extreme_encounter[i]), 2)
if no_enc:
    qprint("no extreme encounters", 2)

### some figures...
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