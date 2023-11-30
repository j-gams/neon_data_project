import numpy as np

from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, Concatenate, Flatten, Reshape, \
    MaxPooling2D

from data_handler import data_wrangler
import math
import pickle

import time

layer_names = ["srtm",
               "nlcd2001",
               "nlcd2004",
               "nlcd2006",
               "nlcd2008",
               "nlcd2011",
               "nlcd2013",
               "nlcd2016",
               "nlcd2019",
               "nlcd2021",
               "aspect",
               "slope",
               "treeage",
               "precip",
               "tempmin",
               "tempmean",
               "tempmax",
               "vapormin",
               "vapormax",
               "ecostress_wue",
               "ecostress_esi",
               "gedi_biomass"]

### (self, rootdir, n_layers, n_folds, cube_dims, batch_size, buffer_nodata, x_ids, y_ids):
### wrangler parameters
rootdir = "../data/pyramid_sets/box_cube_paramadjusted"
n_layers = len(layer_names)
n_folds = 1
#"""
layer_dims = [27,
              27,
              27,
              27,
              27,
              27,
              27,
              27,
              27,
              27,
              27,
              27,
              27,
              27,
              27,
              27,
              27,
              27,
              27,
              15,
              15,
              1]
"""
layer_dims = [34,
              34,
              34,
              34,
              34,
              34,
              34,
              34,
              34,
              34,
              34,
              34,
              1,
              2,
              2,
              2,
              2,
              2,
              2,
              15,
              15,
              1]
"""
batch_size = 32
buffer_nodata = -99
x_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
y_ids = [19, 20, 21]

### compute combinations
unique_layer_dims = []
unique_ydims = []
#layer_cat = []
for i in range(len(layer_dims)):
    if i in x_ids and layer_dims[i] not in unique_layer_dims:
        unique_layer_dims.append(layer_dims[i])
    if i in y_ids and layer_dims[i] not in unique_ydims:
        unique_ydims.append(layer_dims[i])

sort_unique = list(unique_layer_dims)
sort_unique.sort()
unique_freq = [0 for ii in range(len(sort_unique))]
unique_ydims.sort()
unique_yfreq = [0 for ii in range(len(unique_ydims))]
for i in range(len(layer_dims)):
    if i in x_ids:
        for j in range(len(sort_unique)):
            if layer_dims[i] == sort_unique[j]:
                unique_freq[j] += 1
    else:
        for j in range(len(unique_ydims)):
            if layer_dims[i] == unique_ydims[j]:
                unique_yfreq[j] += 1

print(unique_ydims, unique_yfreq)

### training wrangler
tr_wrangler = data_wrangler(rootdir, n_layers, n_folds, layer_dims, batch_size, buffer_nodata, x_ids, y_ids)
### val wrangler
va_wrangler = data_wrangler(rootdir, n_layers, n_folds, layer_dims, batch_size, buffer_nodata, x_ids, y_ids)

save_dir = "model_saves"

n_epochs = 20


class lossCallback(Callback):
    def __init__(self):
        self.logs = dict()
        self.init = False
        self.lasttime = -1

    def on_epoch_end(self, epoch, logs={}):
        keys = list(logs.keys())
        if not self.init:
            for k in keys:
                self.logs[k] = []
            self.init = True
            self.logs["time"] = []
        # ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        # print("CALLBACK: epoch", epoch, "keys:", keys)
        ctime = time.time()
        #if self.lasttime == -1:
        #    ttime = 0
        #else:
        #    ttime = ctime - self.lasttime
        self.logs["time"].append(ctime)
        self.lasttime = ctime
        for k in keys:
            self.logs[k].append(logs.get(k))


def build_model_1(ldims, yoff, uniquedimssorted, freq, ydims_unique, ydims_freq):
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
            ### 34 -> 32 -> 16 -> 8 -> 4
            c1 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="valid")(inputs[i])
            c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
            c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
            c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
            c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
        convs.append(Flatten()(c1))

    merge = Concatenate()(convs)
    c2 = Dense(4000, activation="relu")(merge)
    c2 = Dense(4000, activation="relu")(c2)
    c2 = Dense(2000, activation="relu")(c2)
    c2 = Dense(1200, activation="relu")(c2)

    y1a = Dense((1 ** 2) * 1, activation="relu")(c2)
    y2a = Dense((15 ** 2) * 2, activation="relu")(c2)

    y1b = Reshape((1, 1, 1), name="agb")(y1a)
    y2b = Dense(15 ** 2, activation="relu")(y2a)
    y2c = Reshape((15, 15, 1), name="wue")(y2b)
    y2d = Dense(15 ** 2, activation="relu")(y2a)
    y2e = Reshape((15, 15, 1), name="esi")(y2d)
    yfinal = [y2c, y2e, y1b]

    return Model(inputs=inputs, outputs=yfinal)


for i in range(n_folds):
    tr_wrangler.set_fold(0)
    va_wrangler.set_fold(0)
    tr_wrangler.set_mode("train")
    va_wrangler.set_mode("val")

    model = build_model_1(layer_dims, y_ids[0], sort_unique, unique_freq, unique_ydims, unique_yfreq)
    model.compile(loss="mse")
    print(model.summary())
    print("^^")

    callback1 = lossCallback()
    callbacks = []
    # callback = ModelCheckpoint(save_dir + "/checkpoint.h5",
    #                           monitor="val_mean_squared_error",
    #                           verbose=2,
    #                           mode="min",
    #                           save_best_only=True,
    #                           save_freq="epoch",
    #                           save_weights_only=True)
    callbacks.append(callback1)

    model.fit(tr_wrangler, callbacks=callbacks, epochs=n_epochs, validation_data=va_wrangler, verbose=1)
    model.save_weights(save_dir + "/last_epoch_adjcube_box.h5")
    ### save executable
    ### need to save logs...
    with open(save_dir + "/callbacklog_both.txt", "wb") as myFile:
        pickle.dump(callback1.logs, myFile)
        keylist = list(callback1.logs.keys())
        for k in keylist:
            print(k, callback1.logs[k])

    with open(save_dir + "/callbacklog_both.txt", "rb") as myFile:
        picklerick = pickle.load(myFile)
    print(callback1.logs)
    print()
