print("importing...")
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

#conda activate code_match; cd work/earthlab/munge_data/; python match_create_set.py

### whether to create new data file structure (if one does not exist) and what to name it
create_fs = True
fs_name = "data_nn_gedi"

### reformat data to avoid mega ram issues...
### whether to reformat gedi coordinate data
gencoords = False
### whether to reformat other gedi data
genelse = False
### keywords -- reformat these fields
regen = ["cover"]
print("generating coordinates: " + str(gencoords))
print("generating other fields: " + str(genelse))
if create_fs:
    print("checking for data file structure")
    if not os.path.isdir(fs_name):
        print("creating data file structure")
        os.mkdir(fs_name)
        os.mkdir(fs_name + "/datasrc")
        os.mkdir(fs_name + "/datasrc/x_img")
        os.mkdir(fs_name + "/gedi_reformat")
    if gencoords:
        os.system("rm " + fs_name + "/gedi_reformat/gedi_coords.txt")
        os.system("touch " + fs_name + "/gedi_reformat/gedi_coords.txt")
    if genelse:
        for elt in regen:
            os.system("rm " + fs_name + "/gedi_reformat/gedi_" + elt + ".txt")
            os.system("touch " + fs_name + "/gedi_reformat/gedi_" + elt + "cover"".txt")
    test_img_ = np.zeros((3, 4))
    #arrReshaped = test_img_.reshape(test_img_.shape[0], -1) #see bookmarked page on how to invert this
    np.savetxt(fs_name + "/datasrc/x_img/x_test.csv", test_img_, delimiter=",", newline="\n")

print("done importing packages")
#objective: resample srtm, nlcd to 70x70 for each pixel in ECOSTRESS
print("loading ecostress data...")
#ecos = rxr.open_rasterio("ecostress_WUE/WUE_Median_Composite_AOI.tif", masked=True).squeeze()
ecos_g = gdal.Open("ecostress_WUE/WUE_Median_Composite_AOI.tif")
print("loading nlcd data...")
#nlcd = rxr.open_rasterio("nlcd_raw/clipped_nlcd.tif", masked=True).squeeze()
nlcd_g = gdal.Open("nlcd_raw/clipped_nlcd.tif")
print("loading srtm data...")
#srtm = rxr.open_rasterio("srtm_raw/combined.tif", masked=True).squeeze()
srtm_g = gdal.Open("srtm_raw/combined.tif")

#resolution of ecostress data
#resolution of pixels in data
target_res = 70
xpixel_res = 5
imgsize = target_res // xpixel_res
#img input size is a target_res / xpixel_res square

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

print("loading GEDI data...")
gedi_pts = shapefile.Reader("GEDI_2B_clean/GEDI_2B_clean.shp")
#gedi_g = ogr.Open("GEDI_2B_clean/GEDI_2B_clean.shp")
print("loading GEDI records...")
grecs = gedi_pts.shapeRecords()

print("reformatting grecs...?")
if gencoords:
    print("reformatting coordinate data")
    progress = 0
    for i in range(len(grecs)):
        progress += 1
        if progress % (len(grecs) // 100) == 0:
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
            if progress % (len(grecs) // 100) == 0:
                print("-", end="", flush=True)
            for j in range(len(gedi_pts.fields)):
                if elt in gedi_pts.fields[j][0]:
                    os.system('echo "' + str(grecs[i].record[j]) + ';" >> ' + fs_name + '/gedi_reformat/gedi_' + elt + '.txt')
        print("---> ")
        print("done reformatting " + elt + " data")

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
    gedi_bup = gpd.read_file("GEDI_2B_clean/GEDI_2B_clean.shp")\
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
        if "voronoi" in gedi_mode:


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