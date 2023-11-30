import numpy as np

from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, Concatenate, Flatten, Reshape, MaxPooling2D

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
rootdir = "../data/test_box1"
n_layers = len(layer_names)
n_folds = 1
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

#for i in range(len(unique_layer_dims)):
#    for j in range(len(layer_dims)):
#        if layer_dims[j] == unique_layer_dims[i]:
#            layer_cat.append(i)

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
print("layers", n_layers, len(layer_names), len(layer_dims))
tr_wrangler = data_wrangler(rootdir, n_layers, n_folds, layer_dims, batch_size, buffer_nodata, x_ids, y_ids)
### val wrangler
va_wrangler = data_wrangler(rootdir, n_layers, n_folds, layer_dims, batch_size, buffer_nodata, x_ids, y_ids)

save_dir = "model_saves"

n_epochs = 20

class lossCallback(Callback):
    def __init__ (self):
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
        #['loss', 'accuracy', 'val_loss', 'val_accuracy']
        #print("CALLBACK: epoch", epoch, "keys:", keys)
        ctime = time.time()
        # if self.lasttime == -1:
        #    ttime = 0
        # else:
        #    ttime = ctime - self.lasttime
        self.logs["time"].append(ctime)
        self.lasttime = ctime
        for k in keys:
            self.logs[k].append(logs.get(k))

def build_model_1(ldims, yoff, uniquedimssorted, freq, ydims_unique, ydims_freq):
    xdims = ldims[:yoff]
    ydims = ldims[yoff:]
    inputs = []
    convs1 = [[] for ii in range(len(uniquedimssorted))]
    for i in range(len(xdims)):
        inputs.append(Input(shape=(xdims[i], xdims[i], 1)))
        if xdims[i] == uniquedimssorted[0]:
            ### 1 -> 1
            c1 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1))(inputs[i])
            convs1[0].append(c1)
        elif xdims[i] == uniquedimssorted[1]:
            ### 2 -> 2
            c1 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1))(inputs[i])
            convs1[1].append(c1)
        elif xdims[i] == uniquedimssorted[2]:
            ### 34 -> 17
            c1 = Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="same")(inputs[i])
            convs1[2].append(c1)

    convs2 = [[] for ii in range(len(uniquedimssorted) - 1)]
    for i in range(len(convs1)):
        if i == 0:
            ### 1 -> 1
            convs2[0].append(convs1[0][0])
        else:
            c2 = Concatenate(axis=3)(convs1[i])
            if i == 1:
                ### 2 -> 2
                c2 = Conv2D(filters=8*freq[1], kernel_size=(1, 1), strides=(1, 1))(c2)
            if i == 2:
                ### 17 -> 8
                c2 = Conv2D(filters=8*2*freq[2], kernel_size=(3, 3), strides=(2, 2), padding="valid")(c2)
                ### 8 -> 2
                c2 = Conv2D(filters=8*4*freq[2], kernel_size=(5,5), strides=(4,4), padding="same")(c2)
            convs2[1].append(c2)

    convs3 = []
    for i in range(len(convs2)):
        if i == 0:
            convs3.append(convs2[0][0])
        else:
            c3 = Concatenate(axis=3)(convs2[i])
            ### 2 -> 2
            c3 = Conv2D(filters=8 * 4 * (freq[1] + freq[2]), kernel_size=(1, 1), strides=(1, 1))(c3)
            ### 2 -> 1
            c3 = Conv2D(filters=8 * 8 * (freq[1] + freq[2]), kernel_size=(2, 2), strides=(2, 2))(c3)
            convs3.append(c3)

    ### 1x1x(64*18)
    c4 = Concatenate(axis=3)(convs3)
    c4 = Conv2D(filters=8 * 8 * (freq[1] + freq[2] + freq[0]), kernel_size=(1, 1), strides=(1, 1))(c4)
    c5 = Flatten()(c4)

    c5 = Dense(1600, activation="relu")(c5)
    c5 = Dense(1200, activation="relu")(c5)
    c5 = Dense(1600, activation="relu")(c5)
    ysplit = []
    for j in range(len(ydims_unique)):
        yj = Dense((ydims_unique[j] ** 2) * ydims_freq[j], activation="relu")(c5)
        ysplit.append(yj)
    associated_dim = []
    ysplit2 = []
    for j in range(len(ydims_unique)):
        if ydims_freq[j] == 1:
            yj2 = Reshape((ydims_unique[j], ydims_unique[j], 1))(ysplit[j])
            ysplit2.append(yj2)
            associated_dim.append(ydims_unique[j])
        else:
            for i in range(ydims_freq[j]):
                yj2a = Dense(ydims[j] ** 2, activation="relu")(ysplit[j])
                yj2b = Reshape((ydims_unique[j], ydims_unique[j], 1))(yj2a)
                ysplit2.append(yj2b)
                associated_dim.append(ydims_unique[j])
    yfinal = []
    for i in range(len(ydims)):
        ### find one w/ appropriate exit dim
        for j in range(len(ysplit2)):
            if associated_dim[j] == ydims[i]:
                yfinal.append(ysplit2.pop(j))
                associated_dim.pop(j)
                break
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
    #callback = ModelCheckpoint(save_dir + "/checkpoint.h5",
    #                           monitor="val_mean_squared_error",
    #                           verbose=2,
    #                           mode="min",
    #                           save_best_only=True,
    #                           save_freq="epoch",
    #                           save_weights_only=True)
    callbacks.append(callback1)

    model.fit(tr_wrangler, callbacks=callbacks, epochs=n_epochs, validation_data=va_wrangler, verbose=1)
    model.save_weights(save_dir + "/last_epoch_both.h5")
    ### save executable
    keylist = list(callback1.logs.keys())
    for k in keylist:
        print(k, callback1.logs[k])
    ### need to save logs...
    with open(save_dir + "/callbacklog_both.txt", "wb") as myFile:
        pickle.dump(callback1.logs, myFile)

    with open(save_dir + "/callbacklog_both.txt", "rb") as myFile:
        picklerick = pickle.load(myFile)
    print(callback1.logs)
    print()
