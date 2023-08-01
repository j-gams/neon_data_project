import numpy as np

from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, Concatenate, Flatten, Reshape

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

n_epochs = 10

def build_model_1(ldims, yoff):
    xdims = ldims[:yoff]
    ydims = ldims[yoff:]
    inputs = []
    convs = []
    for i in range(len(xdims)):
        inputs.append(Input(shape=(xdims[i], xdims[i], 1)))
        if xdims[i] < 3:
            c1 = Conv2D(filters=8, kernel_size=(1,1), strides=(1,1))(inputs[i])
            c1 = Conv2D(filters=16, kernel_size=(1,1), strides=(1,1))(c1)
        else:
            c1 = Conv2D(filters=8, kernel_size=(3,3), strides=(3,3))(inputs[i])
            c1 = Conv2D(filters=16, kernel_size=(3,3), strides=(3,3))(c1)
        c1 = Flatten()(c1)
        convs.append(c1)
    c2 = Concatenate(axis=1)(convs)
    c2 = Flatten()(c2)
    c2 = Dense(1000, activation="relu")(c2)
    c2 = Dense(1000, activation="relu")(c2)
    c2 = Dense(1000, activation="relu")(c2)
    ysplit = []
    for j in range(len(ydims)):
        yi = Dense(500, activation="relu")(c2)
        yi = Dense(ydims[j] ** 2, activation="relu")(yi)
        yi = Reshape((ydims[j], ydims[j], 1))(yi)
        ysplit.append(yi)

    return Model(inputs=inputs, outputs=ysplit)

for i in range(n_folds):
    tr_wrangler.set_fold(0)
    va_wrangler.set_fold(0)
    tr_wrangler.set_mode("train")
    va_wrangler.set_mode("val")

    model = build_model_1(layer_dims, 5)
    model.compile(loss="mse")
    model.load_weights(save_dir + "/last_epoch.h5")

    model.fit(tr_wrangler, epochs=n_epochs, validation_data=va_wrangler, verbose=1)