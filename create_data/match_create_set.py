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
### critical fields             [-c comma separated field keywords, optional]
###                             - fields to include in the output samples
### k closest approximation     [-k int, optional (default 10)]
### verbosity                   [-q {0, 1, 2}, optional (default 2, verbose)

### Usage Example:
### python match_create_set.py ../raw_data/srtm_raw/srtm_clipped.tif,../raw_data/nlcd_raw/nlcd_clipped.tif ../raw_data/gedi_pts/GEDI_2B_clean.shp ../raw_data/ecos_wue/WUE_Median_Composite_AOI.tif 70 5 true ../data/data_interpolated --lomem --gencoords --genetc -c cover,pavd,fhd -q 2
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

if len(sys.argv < 8):
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
        elif sys.argv[i] == "-c":
            critical_fields = sys.argv[i+1].split(",")
        elif sys.argv[i] == "-q":
            verbosity = int(sys.argv[i+1])
if not init_ok:
    sys.exit("missing or incorrect command line arguments")
imgsize = y_res // y_pix

def qprint(instr, power):
    if power <= verbosity:
        print(instr)

qprint("importing packages", 2)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
import earthpy as et
import earthpy.plot as ep
from osgeo import gdal
from osgeo import ogr
import shapefile
from longsgis import voronoiDiagram4plg

qprint("done importing packages", 2)

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
                qprint("reformatted coordinate file detected, skipping.")
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

    test_img_ = np.zeros((3, 4))
    # arrReshaped = test_img_.reshape(test_img_.shape[0], -1) #see bookmarked page on how to invert this
    np.savetxt(fs_name + "/datasrc/x_img/x_test.csv", test_img_, delimiter=",", newline="\n")

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
    layernames.append(tdataname)

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
yrsize = (yraster.RasterXSize, yraster.RasterYSize)
yulh, ypxh, _, yulv, _, ypxv =yraster.GetGeoTransform()
ypxv = abs(ypxv)
### print crs info
y_npar = yraster.ReadAsArray.transpose()


xpoints = []
ptlayers = []
ptindexer = {}
grecs = []
qprint("loading shape-point data", 1)
for pt in point_shape:
    tdataname = pt.split("/")[-1]
    qprint("loading " + tdataname + " data...", 2)
    xpoints.append(shapefile.Reader(pt))
    qprint("loading " + tdataname + " records...", 2)
    grecs.append(xpoints[-1].shapeRecords())

    for i in range(len(critical_fields)):
        ptl_idx = 0
        for j in range(len(xpoints[-1].fields)):
            if critical_fields[i] in xpoints[-1].fields[j][0]:
                ### given the layer (key) provides the ptfile #, critical field #, field index within that shape,
                ### and id within the np array (if exists)
                ptindexer[len(ptlayers)] = (len(xpoints) -1, i, j, ptl_idx)
                ptlayers.append(critical_fields[i] + "_" + str(ptl_idx))
                ptl_idx += 1


    if lo_mem:
        if gen_coords and not found_coord:
            qprint("reformatting shapefile coordinates. This could take a while...", 1)
            if verbosity > 0:
                print("progress 0/50 ", end="", flush=True)
            progress = 0
            for i in range(len(grecs[-1])):
                progress += 1
                if verbosity > 0 and progress % (len(grecs[-1]) // 50) == 0:
                    print("-", end="", flush=True)
                a, b = grecs[-1][i].records[3], grecs[-1][i].record[2]
                os.system('echo "' + str(a) + ',' + str(b) + '\n" >> ' + fs_loc + '/point_reformat/pt_coords.txt')

            qprint("---> 50/50 (done)", 1)

        for i in range(len(critical_fields)):
            if gen_etc and not found_etc[i]:
                qprint("reformatting critical field " + critical_fields[i] + ". This could take a while...", 1)
                if verbosity > 0:
                    print("progress 0/50 ", end="", flush=True)
                progress = 0
                for j in range(len(grecs[-1])):
                    progress += 1
                    if verbosity > 0 and progress % (len(grecs[-1]) // 50) == 0:
                        print("-", end="", flush=True)
                    dumpstr = ""
                    for k in range(len(xpoints[-1].fields)):
                        if critical_fields[i] in xpoints[-1].fields[k][0]:
                            dumpstr += str(grecs[-1][j].record[k]) + ","
                    os.system('echo "' + dumpstr[:-1] + ';" >> ' + fs_loc + '/point_reformat/pt_' + critical_fields[i] + '.txt')
                print("---> ")
                print("done reformatting " + critical_fields[i] + " data")
        del xpoints[-1]
        del grecs[-1]

### now try to load the data back in
if lo_mem:
    ### clear hi-mem data (moved to above)

    ### load data from file
    crit_npar = []
    for i in range(len(critical_fields)):
        ##load in
        crit_npar.append(np.genfromtxt( + '/point_reformat/pt_' + critical_fields[i] + '_coords.txt', delimiter=','))

### now we have all the data loaded in?

def getter(layer, index):
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
    return (x1 - x2) ** 2 + (y1 - y2) ** 2

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

### ok now actually build the data
pr_unit = (yrsize[0] * yrsize[1]) // 50
qprint("each step represents " + int(pr_unit) + " samples generated")
progress = 0
nsuccess = 0
database = []
channels = len(xr_npar) + len(ptlayers)
if verbosity > 0:
    print("progress 0/50 ", end="", flush=True)
for i in range(yrsize[0]):
    for j in range(yrsize[1]):
        progress += 1
        if verbosity > 0 and progress % pr_unit == 0:
            print("-", end="", flush=True)
        if y_npar[i, j] != yndv:
            x_img = np.zeros((channels, imgsize, imgsize))
            database.append([nsuccess, y_npar[i, j]])
            for k in range(len(xr_npar)):# in xr_npar:
                for si in range(imgsize):
                    for sj in range(imgsize):
                        sxoffset = (2 * si + 1) / (2 * imgsize)
                        syoffset = (2 * sj + 1) / (2 * imgsize)
                        # convert index to coordinates with y raster crs
                        # then convert back to index with x raster crs
                        tempx, tempy = idx_pixctr(i+sxoffset, j+syoffset, xr_params[k][0], xr_params[k][1],
                                                  xr_params[k][2], xr_params[k][3], mode='cst')
                        tempi, tempj = coords_idx(tempx, tempy, xr_params[k][0], xr_params[k][1],
                                                  xr_params[k][2], xr_params[k][3])

                        # x_img[si, sj, 0] = srtm_npar[srtm_i, srtm_j]
                        x_img[0, si, sj] = xr_npar[k][tempi, tempj]
            for m in range(len(ptlayers)):


### TODO NOT REFACTORED YET

print("reformatting grecs...?")
if gencoords:
    print("reformatting coordinate data")
    progress = 0
    for i in range(len(grecs)):
        progress += 1
        if progress % (len(grecs) // 50) == 0:
            print("-", end="", flush=True)
        a, b = grecs[i].record[3], grecs[i].record[2]
        os.system('echo "' + str(a) + ',' + str(b) + '\n" >> ' + fs_name + '/gedi_reformat/gedi_coords.txt')
    print("---> ")
    print("done reformatting coordinate data")

if genelse:
    for elt in regen:
        progress = 0
        print("reformatting " + elt + " data")
        for i in range(len(grecs)):
            progress += 1
            if progress % (len(grecs) // 50) == 0:
                print("-", end="", flush=True)
            dumpstr = ""
            for j in range(len(gedi_pts.fields)):
                if elt in gedi_pts.fields[j][0]:
                    dumpstr += str(grecs[i].record[j]) + ","
            os.system('echo "' + dumpstr[:-1] + ';" >> ' + fs_name + '/gedi_reformat/gedi_' + elt + '.txt')
        print("---> ")
        print("done reformatting " + elt + " data")

sys.exit("wank")
print("done reformatting grecs.")
print("loading gedi coords...")
del grecs

#with open(fs_name + '/gedi_reformat/gedi_coords.txt', 'r', newline=';') as ff1:
#    gedi_pts_npar = np.loadtxt(ff1, dtype='float', usecols=(0,1))
gedi_pts_npar = np.genfromtxt(fs_name + '/gedi_reformat/gedi_coords.txt', delimiter=',')
print(gedi_pts_npar)
n_gedi = len(gedi_pts_npar)
print("gedi coords shape:", gedi_pts_npar.shape)

print("*** data info ***")
### get ecostress data info
ecos_rband = ecos_g.GetRasterBand(1)
ecos_ndv = ecos_rband.GetNoDataValue()
ecos_rsize = (ecos_g.RasterXSize, ecos_g.RasterYSize)
ecos_UL_h, ecos_h_spac, _, ecos_UL_v, _, ecos_v_spac = ecos_g.GetGeoTransform()
#account for negative pixel size:
ecos_v_spac = abs(ecos_v_spac)
print("ECOSTRESS crs info:", ecos_UL_h, ecos_UL_v, ecos_h_spac, ecos_v_spac)
print("ECOSTRESS raster size:", ecos_rsize)
print("ECOSTRESS no data value:", ecos_ndv)
ecos_npar = ecos_g.ReadAsArray().transpose()
print(ecos_npar.shape)

### get srtm data info
srtm_rband = srtm_g.GetRasterBand(1)
srtm_ndv = srtm_rband.GetNoDataValue()
srtm_rsize = (srtm_g.RasterXSize, srtm_g.RasterYSize)
srtm_UL_h, srtm_h_spac, _, srtm_UL_v, _, srtm_v_spac = srtm_g.GetGeoTransform()
#account for negative pixel size:
srtm_v_spac = abs(srtm_v_spac)
print("SRTM crs info:", srtm_UL_h, srtm_UL_v, srtm_h_spac, srtm_v_spac)
print("SRTM raster size:", srtm_rsize)
print("SRTM no data value:", srtm_ndv)
srtm_npar = srtm_g.ReadAsArray().transpose()

### get srtm data info
nlcd_rband = nlcd_g.GetRasterBand(1)
nlcd_ndv = nlcd_rband.GetNoDataValue()
nlcd_rsize = (nlcd_g.RasterXSize, nlcd_g.RasterYSize)
nlcd_UL_h, nlcd_h_spac, _, nlcd_UL_v, _, nlcd_v_spac = nlcd_g.GetGeoTransform()
#account for negative pixel size:
nlcd_v_spac = abs(nlcd_v_spac)
print("NLCD crs info:", nlcd_UL_h, nlcd_UL_v, nlcd_h_spac, nlcd_v_spac)
print("NLCD raster size:", nlcd_rsize)
print("NLCD no data value:", nlcd_ndv)
nlcd_npar = nlcd_g.ReadAsArray().transpose()

### test transformations
print("testing transformations:")
tix, tiy = 0, 0
print(tix, tiy)
tcx, tcy = idx_pixctr(tix, tiy, ecos_UL_h, ecos_UL_v, ecos_h_spac, ecos_v_spac, mode='ctr')
print(tcx, tcy)
tix, tiy = coords_idx(tcx, tcy, ecos_UL_h, ecos_UL_v, ecos_h_spac, ecos_v_spac)
print(tix, tiy)

#channels: srtm (0), nlcd (1), gedi(2)
channels = 3
print("generating data...")
print("constructing X,y pairs...")
progress = 0
steps = 100
outta = (ecos_rsize[0] * ecos_rsize[1]) // steps

"""test
### test
print("setting diagnostic tiles:")
print(nlcd_v_spac/ecos_v_spac)
test_x, test_y = idx_pixctr(0, 0, ecos_UL_h, ecos_UL_v, ecos_h_spac, ecos_v_spac)
test_i, test_j = coords_idx(test_x, test_y, srtm_UL_h, srtm_UL_v, srtm_h_spac, srtm_v_spac)
print(test_x, test_y)
print(test_i, test_j)
ecos_npar[0, 0] = 5
srtm_npar[test_i, test_j] = 5
srtm_npar[test_i+1, test_j] = 10
srtm_npar[test_i+2, test_j] = 0
srtm_npar[test_i,test_j+1] = 4
srtm_npar[test_i+1, test_j+1] = 8
srtm_npar[test_i+2, test_j+1] = 8
srtm_npar[test_i,test_j+2] = 5
srtm_npar[test_i+1,test_j+2]  =2
srtm_npar[test_i+2, test_j+2] = 10
print(srtm_npar[test_i:test_i+3, test_j:test_j+3])
"""
print("ecos_UL in srtm")
kclose = 10
printall = True
gotone = False
nsuccess = 0
gedi_mode = ["voronoi", "nn"]
if "voronoi" in gedi_mode:
    print("loading points in geopandas")
    #need to load with geopandas...
    gedi_bup = gpd.read_file("GEDI_2B_clean/GEDI_2B_clean.shp")
    bdry_wrap = shapefile.Writer("tpath")
    bdry_wrap.poly()

for i in range(ecos_rsize[0]):
    for j in range(ecos_rsize[1]):
        progress += 1
        if progress % outta == 0:
            print("-", end="", flush=True)
        if ecos_npar[i, j] == ecos_ndv:
            if i == 0 and j == 0:
                print(ecos_npar[i, j])
            continue
        #x_img = np.zeros((imgsize, imgsize, channels))
        x_img = np.zeros((channels, imgsize, imgsize))
        yval = ecos_npar[i, j]
        #do srtm (ch0)
        for si in range(imgsize):
            for sj in range(imgsize):
                sxoffset = (2*si + 1)/(2*imgsize)
                syoffset = (2*sj + 1)/(2*imgsize)
                # convert index to coordinates with ecostress crs
                # then convert back to index with srtm crs
                srtm_x, srtm_y = idx_pixctr(i+sxoffset, j+syoffset, ecos_UL_h, ecos_UL_v, ecos_h_spac, ecos_v_spac,
                                            mode='cst')
                srtm_i, srtm_j = coords_idx(srtm_x, srtm_y, srtm_UL_h, srtm_UL_v, srtm_h_spac, srtm_v_spac)
                #x_img[si, sj, 0] = srtm_npar[srtm_i, srtm_j]
                x_img[0, si, sj] = srtm_npar[srtm_i, srtm_j]
        #do nlcd (ch1)
        for si in range(imgsize):
            for sj in range(imgsize):
                sxoffset = (2 * si + 1) / (2 * imgsize)
                syoffset = (2 * sj + 1) / (2 * imgsize)
                # convert index to coordinates with ecostress crs
                # then convert back to index with nlcd crs
                nlcd_x, nlcd_y = idx_pixctr(i+sxoffset, j+syoffset, ecos_UL_h, ecos_UL_v, ecos_h_spac, ecos_v_spac,
                                            mode='cst')
                nlcd_i, nlcd_j = coords_idx(nlcd_x, nlcd_y, nlcd_UL_h, nlcd_UL_v, nlcd_h_spac, nlcd_v_spac)
                #x_img[si, sj, 1] = nlcd_npar[nlcd_i, nlcd_j]
                x_img[1, si, sj] = nlcd_npar[nlcd_i, nlcd_j]
        if "nn" in gedi_mode:
            #get k closest to center of 70m pixel
            kn_ids = getkclose(gedi_pts_npar, i, j, kclose, ecos_UL_h, ecos_UL_v, ecos_h_spac, ecos_v_spac)
            for si in range(imgsize):
                for sj in range(imgsize):
                    sxoffset = (2 * si + 1) / (2 * imgsize)
                    syoffset = (2 * sj + 1) / (2 * imgsize)
                    gedi_x, gedi_y = idx_pixctr(i+sxoffset, j+syoffset, ecos_UL_h, ecos_UL_v, ecos_h_spac, ecos_v_spac,
                                                mode='cst')
                    #find kn_id pt closest to center of pixel
                    mindist = 180
                    minpt = None
                    #a = grecs[i].record[3]
                    #b = grecs[i].record[2]
                    for k in range(kclose):
                        tdist = cdist(gedi_pts_npar[kn_ids[k], 0], gedi_pts_npar[kn_ids[k], 1], gedi_x, gedi_y)
                        if tdist < mindist:
                            mindist = tdist
                            minpt = kn_ids[k]
                    #get value of nearest elt
                    # x_img[si, sj, 1] = nlcd_npar[nlcd_i, nlcd_j]
                    x_img[2, si, sj] = minpt #TEMPORARY#nlcd_npar[nlcd_i, nlcd_j]
        #if "voronoi" in gedi_mode:


        #plot image
        if printall and False:
            print("diagnostics:")
            print("i, j: ", i, j)
            print("offset base", ecos_h_spac/imgsize)
            test_x, test_y = idx_pixctr(i, j, ecos_UL_h, ecos_UL_v, ecos_h_spac, ecos_v_spac)
            test_i, test_j = coords_idx(test_x, test_y, srtm_UL_h, srtm_UL_v, srtm_h_spac, srtm_v_spac)
            print(srtm_npar[test_i:test_i+3, test_j:test_j+3])
            print(x_img[0])
            print(yval)
            print(x_img.shape, x_img.dtype, imgsize, channels)
            printall = False
            print(x_img[10000000])
        if len(np.unique(x_img[1])) > 2 and False:
            plt.imshow(x_img[2])
            plt.show()
        #save file
        np.savetxt(fs_name + "/datasrc/x_img/x_" + str(nsuccess) +".csv", x_img.reshape(x_img.shape[0],-1), delimiter=",")
        #load with loaded_array.reshape(loadedArr.shape[0], loadedArr.shape[1]//arr.shape[2], arr.shape[2]
        nsuccess += 1
print()
print("done")
