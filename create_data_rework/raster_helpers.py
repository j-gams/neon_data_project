import numpy as np
from osgeo import gdal

def save_raster(prefix, filename, rasterdata, rastercrs, rasterproj, rasternodata):

    driver = gdal.GetDriverByName("GTiff")
    outname = prefix + "/" + filename + ".tif"
    layer_out = driver.Create(outname, rasterdata.shape[0], rasterdata.shape[1], 1, gdal.GDT_Float32)
    layer_out.SetGeoTransform(rastercrs)
    layer_out.SetProjection(rasterproj)
    ### save layer geotif
    layer_out.GetRasterBand(1).WriteArray(rasterdata.transpose())
    layer_out.GetRasterBand(1).SetNoDataValue(rasternodata)
    layer_out.FlushCache()