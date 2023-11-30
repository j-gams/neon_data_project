### Written by Jerry Gammie @j-gams
import sys
import math
import numpy as np
import align_create_helpers as ach
import multiprocessing
import os
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import h5py
from multiprocessing import Pool

from rfdata_loader import rfloader, piloader

gdal.UseExceptions()
### most recent?

### python create_test.py ../data/raster/nlcd2011_clipped_co_reproj.tif ../data/raster/ecostresswue_clipped_co.tif ../data/set/data_h5test

### python align_create.py ../data/raster/treeage_clipped_co_reproj.tif ../data/point/GEDI_2B_clean.shp ../data/raster/ecostressesi_clipped_co.tif 70 5 true ../data/set/data_h5test --lomem --gencoords --genetc --override --prescreen --h5mode=h5 --cfields=cover,pavd,fhd --orient=hwc --pad=1 --hashpad=10 --chunk=10  --q=2
### python align_create.py ../data/raster/srtm_clipped_co.tif,../data/raster/nlcd_clipped_co_reproj.tif,../data/raster/slope_clipped_co.tif,../data/raster/aspect_clipped_co.tif ../data/point/GEDI_2B_clean.shp ../data/raster/WUE_Median_Composite_AOI.tif 70 5 true ../data/set/data_h5test --lomem --gencoords --genetc --override --prescreen --h5mode=h5 --cfields=cover,pavd,fhd --orient=hwc --pad=1 --hashpad=10 --chunk=10  --q=2

def mpdispatch(inparams):
    bands, xrnum, params = inparams
    res = []
    for elt in bands:
        res.append(ach.badmpalign(params, xrnum, elt))
    return res

def chunk_list(nelts, nchunks, idx):
    size = math.ceil(nelts / nchunks)
    return [ii for ii in range(idx*size, min((idx + 1)*size, nelts))]

if __name__ == "__main__":


    x_raster_locs = sys.argv[1].split(",")
    y_raster_loc = sys.argv[2]
    fs_loc = sys.argv[3]
    ### load x raster
    xraster = []
    xr_crs = []
    layernames = []
    ndv_vals = []
    xr_rsize = []
    xr_params = []
    xr_npar = []

    stopstep = -1
    skip_save = False

    print("loading raster data", 1)
    for loc in x_raster_locs:
        tdataname = loc.split("/")[-1]
        print("loading " + tdataname + " data...", 2)
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
    print("loading " + tyname + " data...", 2)
    yraster = gdal.Open(y_raster_loc)
    yrband = yraster.GetRasterBand(1)
    yndv = yrband.GetNoDataValue()
    print("no data value", yndv)
    yrsize = (yraster.RasterXSize, yraster.RasterYSize)
    print("y raster dimensions: " + str(yrsize), 2)
    yulh, ypxh, _, yulv, _, ypxv = yraster.GetGeoTransform()
    ypxv = abs(ypxv)
    ### print crs info
    yr_crs = yraster.GetProjection()
    print(yulh, yulv, ypxh, ypxv)
    y_npar = yraster.ReadAsArray().transpose()

    if stopstep == 2:
        print("breaking at point 2")
        sys.exit(0)

    ### TODO --- make this 16x16 instead of 14x14... add 1 unit of buffer on each side
    pr_unit = (yrsize[0] * yrsize[1]) // 50
    print("each step represents " + str(pr_unit) + " samples generated", 1)
    progress = 0
    nsuccess = 0
    # extreme_encounter = [0 for ii in range(len(xr_npar) + len(ptlayers) + 2)]
    database = []
    # channels = len(xr_npar) + len(ptlayers) + 2
    pd_colnames = ["filename", "y_value", "file_index", "yraster_x", "yraster_y", "avg_mid_dist"]
    landmark_x, landmark_y = ach.coords_idx(-104.876653, 41.139535, yulh, yulv, ypxh, ypxv)
    if skip_save:
        print("warning: running in skip save mode")

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
                      [xr_params[3], xr_crs[3]],
                      [xr_params[4], xr_crs[4]],
                      [xr_params[5], xr_crs[5]],
                      [xr_params[6], xr_crs[6]],
                      [xr_params[7], xr_crs[7]],
                      [xr_params[8], xr_crs[8]],
                      [xr_params[9], xr_crs[9]],
                      [xr_params[10], xr_crs[10]]]
    else:
        x_crs_pack = [[xr_params[0], xr_crs[0]]]
        # [xr_params[1], xr_crs[1]]]

    if False:
        lads = [[0, ach.alignment_sampling, 30, "SRTM_30_globalUL", ("ul")]]
        # [2, ach.alignment_sampling, 30, "Slope_30_globalUL", ("ul")],
        # [3, ach.alignment_sampling, 30, "Aspect_30_globalUL", ("ul")],
        # [1, ach.alignment_sampling, 30, "NLCD_30_globalUL", ("ul")]]
        # [0, ach.alignment_sampling, 10, "SRTM_10_globalUL", ("ul")],
        # [1, ach.alignment_sampling, 10, "Slope_10_globalUL", ("ul")],
        # [2, ach.alignment_sampling, 10, "Aspect_10_globalUL", ("ul")],
        # [3, ach.alignment_sampling, 10, "NLCD_10_globalUL", ("ul")]]
    elif False:
        lads = [[0, ach.alignment_average, 70, "SRTM_70_mean", ("mean", 10, 30)],
                [2, ach.alignment_average, 70, "Slope_70_mean", ("mean", 10, 30)],
                [3, ach.alignment_average, 70, "Aspect_70_mean", ("mean", 10, 30)],
                [1, ach.alignment_average, 70, "NLCD_70_mode", ("mode", 10, 30)],
                [4, ach.alignment_average, 70, "tempmin", ("mean", 10, 800)],
                [5, ach.alignment_average, 70, "tempmax", ("mean", 10, 800)],
                [6, ach.alignment_average, 70, "tempmean", ("mean", 10, 800)],
                [7, ach.alignment_average, 70, "vapormin", ("mean", 10, 800)],
                [8, ach.alignment_average, 70, "vapormax", ("mean", 10, 800)],
                [9, ach.alignment_average, 70, "precip", ("mean", 10, 800)],
                [10, ach.alignment_average, 70, "treeage", ("mean", 10, 1000)]]

    else:
        lads = [  # [0, ach.alignment_average, 70, "NLC2001_70_mode", ("mode", 10, 30)],
            # [1, ach.alignment_average, 70, "NLCD2004_70_mode", ("mode", 10, 30)],
            # [2, ach.alignment_average, 70, "NLCD2006_70_mode", ("mode", 10, 30)],
            # [3, ach.alignment_average, 70, "NLCD2008_70_mode", ("mode", 10, 30)],
            # [4, ach.alignment_average, 70, "NLCD2011_70_mode", ("mode", 10, 30)],
            # [0, ach.alignment_average, 70, "NLCD2013_70_mode", ("mode", 10, 30)],
            [0, ach.alignment_average, 70, "NLCD2001_70_mode", ("mode", 10, 30)]]
        # [7, ach.alignment_average, 70, "NLCD2021_70_mode", ("mode", 10, 30)]]

    print("available cpus", multiprocessing.cpu_count())
    print("xr_npar", xr_npar[0].shape)


    collection_name = "geotifs_sampled_4_esi"
    prefix = "../data/" + collection_name
    os.system("mkdir " + prefix)

    driver = gdal.GetDriverByName("GTiff")

    ### deal with multiprocessing
    for i in range(len(lads)):
        ### initialize aligner (below...?)

        ### deal with multiprocessing

        tgrid_res = lads[i][2]
        texpect_res = lads[i][4][2]
        tbase_res = 70
        ndignore = ndv_vals[lads[i][0]]
        ndoops = -99999
        avg_method = lads[i][4][0]

        dsampler = (tgrid_res // texpect_res) + 2
        brfrac = tgrid_res/tbase_res
        tfrac = tgrid_res / texpect_res
        total_m = (yrsize[0] * tbase_res, yrsize[1] * tbase_res)
        grid_size = ((total_m[0] // tgrid_res) + 1, (total_m[1] // tgrid_res) + 1)
        print("GRID SHAPE", grid_size)

        const_params = [grid_size, dsampler, brfrac, y_crs_pack[0], x_crs_pack[lads[i][0]][0],
                        tfrac, ndignore, ndoops, avg_method]
        #grid_size[0]
        paramarr = []
        ncpu = multiprocessing.cpu_count()
        for j in range(ncpu):
            paramarr.append([chunk_list(grid_size[0], ncpu, j), np.array(xr_npar[lads[i][0]]), const_params])
        with Pool(None) as mpool:
            resarr = mpool.map(mpdispatch, paramarr)

        all_oobc = 0
        just_oobc = 0
        trueres = []
        for outerarr in resarr:
            #print(type(outerarr))
            for elt in outerarr:
                #print(type(elt))
                innerarr, oobitem, anoob = elt
                trueres.append(innerarr)
                all_oobc += oobitem
                just_oobc += anoob
            #trueres.exten(arr1)
        print("all oobs", all_oobc)
        print("any oobs", just_oobc)



        layer_geotif = lads[i][1](lads[i][2], (yrsize[0], yrsize[1]), 70, y_crs_pack,
                                        x_crs_pack[lads[i][0]], lads[i][4], ndv_vals[lads[i][0]])
        layer_geotif.mpimport(trueres)

        ### obtain layer geotifs
        driver = gdal.GetDriverByName("GTiff")
        outname = prefix + "/" + lads[i][3] + ".tif"
        layer_out = driver.Create(outname, layer_geotif.data.shape[0], layer_geotif.data.shape[1], 1, gdal.GDT_Float32)
        layer_out.SetGeoTransform(layer_geotif.newcrs)
        layer_out.SetProjection(layer_geotif.newproj)
        ### save layer geotif
        layer_out.GetRasterBand(1).WriteArray(layer_geotif.data.transpose())
        layer_out.GetRasterBand(1).SetNoDataValue(layer_geotif.nodata_oops)
        layer_out.FlushCache()

    """for i in range(len(lads)):
        ### initialize aligner
        layer_geotif = lads[i][1](lads[i][2], (yrsize[0], yrsize[1]), 70, y_crs_pack,
                                  x_crs_pack[lads[i][0]], lads[i][4], ndv_vals[lads[i][0]])
        ### perform alignment
        layer_geotif.alignbasic(xr_npar[lads[i][0]], subset=100)

        ### obtain layer geotifs
        driver = gdal.GetDriverByName("GTiff")
        outname = prefix + "/" + lads[i][3] + ".tif"
        layer_out = driver.Create(outname, layer_geotif.data.shape[0], layer_geotif.data.shape[1], 1, gdal.GDT_Float32)
        layer_out.SetGeoTransform(layer_geotif.newcrs)
        layer_out.SetProjection(layer_geotif.newproj)
        ### save layer geotif
        layer_out.GetRasterBand(1).WriteArray(layer_geotif.data.transpose())
        layer_out.GetRasterBand(1).SetNoDataValue(layer_geotif.nodata_oops)
        layer_out.FlushCache()

        print("saved to geotif" + outname)"""
