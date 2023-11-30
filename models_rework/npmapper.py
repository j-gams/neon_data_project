
import numpy as np
from osgeo import gdal
from data_handler import data_wrangler

layer_names = ["srtm",
               "nlcd",
               "aspect",
               "slope",
               "treeage",
               "ecostress_esi",
               "gedi_biomass"]

### (self, rootdir, n_layers, n_folds, cube_dims, batch_size, buffer_nodata, x_ids, y_ids):
### wrangler parameters
rootdir = "../data/test_2"
n_layers = 7
n_folds = 1
layer_dims = [34,
              34,
              34,
              34,
              1,
              15,
              1]
batch_size = 32
buffer_nodata = -99
x_ids = [0, 1, 2, 3, 4]
y_ids = [5, 6]

###

tst_wrangler = data_wrangler(rootdir, n_layers, n_folds, layer_dims, batch_size, buffer_nodata, x_ids, y_ids)
save_dir = "model_saves"
legit_coords = np.genfromtxt("../data/test_box1/legal_ids.csv", delimiter=',')
tst_ids = tst_wrangler.test_index


tst_wrangler.set_mode("test")

layer_raster = gdal.Open("../data/raster/gedi_agforestbiomass_clipped_co.tif")
rasterband = layer_raster.GetRasterBand(1)
layer_nd = rasterband.GetNoDataValue()
layer_s = (layer_raster.RasterXSize, layer_raster.RasterYSize)
gtf = layer_raster.GetGeoTransform()
gproj = layer_raster.GetProjection()
#tpxv = abs(tpxv)
layer_data = layer_raster.ReadAsArray().transpose()

del rasterband
del layer_raster

holds = []
for i in range(len(tst_wrangler)):
    hold = tst_wrangler[i][1][1]
    for elt1 in hold:
        holds.append(elt1)

holds = np.array(holds)
print(holds.shape)

empty = np.zeros(layer_data.shape)
empty.fill(float("nan"))

for i in range(len(tst_ids)):
    tempc = legit_coords[tst_ids[i].astype(int)]
    #print(tempc)
    empty[int(tempc[0]), int(tempc[1])] = holds[i]



driver = gdal.GetDriverByName("GTiff")
outname = "msemap1.tif"
layer_out = driver.Create(outname, empty.shape[0], empty.shape[1], 1, gdal.GDT_Float32)
layer_out.SetGeoTransform(gtf)
layer_out.SetProjection(gproj)
layer_out.GetRasterBand(1).WriteArray(empty.transpose())
layer_out.FlushCache()