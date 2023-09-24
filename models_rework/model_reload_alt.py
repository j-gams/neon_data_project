import numpy as np

from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, Concatenate, Flatten, Reshape, MaxPooling2D
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

### training wrangler
tr_wrangler = data_wrangler(rootdir, n_layers, n_folds, layer_dims, batch_size, buffer_nodata, x_ids, y_ids)
### val wrangler
va_wrangler = data_wrangler(rootdir, n_layers, n_folds, layer_dims, batch_size, buffer_nodata, x_ids, y_ids)

save_dir = "model_saves"

n_epochs = 0

def build_model_1(ldims, yoff):
    xdims = ldims[:yoff]
    ydims = ldims[yoff:]
    inputs = []
    convs = []
    for i in range(len(xdims)):
        inputs.append(Input(shape=(xdims[i], xdims[i], 1)))
        if xdims[i] < 3:
            c1 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1))(inputs[i])
            #c1 = Conv2D(filters=16, kernel_size=(1,1), strides=(1,1))(c1)
        else:
            c1 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="valid")(inputs[i])
            c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
            c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
            c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
            c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
        convs.append(Flatten()(c1))

    merge = Concatenate()(convs)
    c2 = Dense(1200, activation="relu")(merge)
    c2 = Dense(1000, activation="relu")(c2)
    c2 = Dense(1600, activation="relu")(c2)
    ysplit = []
    for j in range(len(ydims)):
        yi = Dense(ydims[j] ** 2, activation="relu")(c2)
        ysplit.append(yi)
    y1a = Reshape((ydims[1], ydims[1], 1))(ysplit[1])
    y1b = Concatenate(axis=1)([ysplit[1], ysplit[0]])
    y1c = Dense(225, activation="relu")(y1b)
    y1d = Reshape((ydims[0], ydims[0], 1))(y1c)
    return Model(inputs=inputs, outputs=[y1d, y1a])

for i in range(n_folds):
    tr_wrangler.set_fold(0)
    va_wrangler.set_fold(0)
    tr_wrangler.set_mode("train")
    va_wrangler.set_mode("val")

    model = build_model_1(layer_dims, 5)
    model.compile(loss="mse")
    model.load_weights(save_dir + "/last_epoch_both.h5")

    #model.fit(tr_wrangler, epochs=n_epochs, validation_data=va_wrangler, verbose=1)
    #model.save_weights(save_dir + "/last_epoch.h5")

    ### make some predictions...
    if True:
        legit_coords = np.genfromtxt("../data/test_2/legal_ids.csv", delimiter=',')
        val_ids = va_wrangler.val_ids[0]
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

        empty = np.zeros(layer_data.shape)
        empty.fill(float("nan"))

        ### hopefully only agb
        yhats = []
        holds = []
        for i in range(len(va_wrangler)):
            hold = va_wrangler[i][1][1]
            #holds.append(hold.flatten())
            tr = model(va_wrangler[i][0])[1]
            for elt in tr:
                yhats.append(elt)
            for elt1 in hold:
                holds.append(elt1)
        yhats = np.array(yhats)
        holds = np.array(holds)

        mse = (yhats[:,:,:,0] - holds) ** 2

        print(yhats.shape)
        print(holds.shape)

        ### make plot
        from matplotlib import pyplot as plt
        #yhats
        #
        ##### holds

        plt.scatter(holds.flatten(), yhats.flatten())
        plt.savefig('scatter_y_yhat.png')
        #plt.clear()
        #plt.scatter(yhats.flatten(), mse.flatten())
        #plt.savefig('scatter_y_mse.png')
        #plt.scatter(holds.flatten(), mse.flatten())
        #plt.savefig('scatter_holds_mse.png')

        """
        import pandas as pd

        print(val_ids.shape)
        print(val_ids[:5])
        #val_ids
        for i in range(len(val_ids)):
            tempc = legit_coords[val_ids[i].astype(int)]
            print(tempc)
            empty[int(tempc[0]), int(tempc[1])] = mse[i]

        nanframe = pd.DataFrame(empty)
        nanframe.interpolate(method='nearest')

        empty = nanframe.to_numpy()

        driver = gdal.GetDriverByName("GTiff")
        outname = "msemap1.tif"
        layer_out = driver.Create(outname, empty.shape[0], empty.shape[1], 1, gdal.GDT_Float32)
        layer_out.SetGeoTransform(gtf)
        layer_out.SetProjection(gproj)
        layer_out.GetRasterBand(1).WriteArray(empty.transpose())
        layer_out.FlushCache()
        """