import numpy as np

from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, Concatenate, Flatten, Reshape, MaxPooling2D
from tensorflow.keras.callbacks import Callback
from data_handler import data_wrangler

import time

layer_names = ["srtm",
               "nlcd", 
               "aspect",
               "slope", 
               "treeage",
               "gedi_biomass"]

### (self, rootdir, n_layers, n_folds, cube_dims, batch_size, buffer_nodata, x_ids, y_ids):
### wrangler parameters
rootdir = "../data/test_2a"
n_layers = 6
n_folds = 1
layer_dims = [34,
              34,
              34,
              34,
              1,
              1]
batch_size = 32
buffer_nodata = -99
x_ids = [0, 1, 2, 3, 4]
y_ids = [5]

### training wrangler
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
        if self.lasttime == -1:
            ttime = 0
        else:
            ttime = ctime - self.lasttime
        self.logs["time"].append(ttime)
        self.lasttime = ctime
        for k in keys:
            self.logs[k].append(logs.get(k))

def build_model_1(ldims, yoff):
    xdims = ldims[:yoff]
    ydims = ldims[yoff:]
    inputs = []
    convs = []
    for i in range(len(xdims)):
        inputs.append(Input(shape=(xdims[i], xdims[i], 1)))
        if xdims[i] < 3:
            c1 = Conv2D(filters=8, kernel_size=(1,1), strides=(1,1))(inputs[i])
            #c1 = Conv2D(filters=16, kernel_size=(1,1), strides=(1,1))(c1)
        else:
            c1 = Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding="valid")(inputs[i])
            c1 = Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), padding="same")(c1)
        convs.append(c1)
    
    ## 16x16x16*4
    c1a = Concatenate(axis=3)([convs[0], convs[1], convs[2], convs[3]])
    ### 8x8x16*8
    c1b = Conv2D(filters=16*8, kernel_size=(5,5), strides=(2,2), padding="same")(c1a)
    c1b = MaxPooling2D((2, 2))(c1b)
    c1c = Flatten()(c1b)
    c1d = Flatten()(convs[4])
    c2 = Concatenate(axis=1)([c1d, c1c])
    c2 = Flatten()(c2)
    c2 = Dense(1600, activation="relu")(c2)
    c2 = Dense(1200, activation="relu")(c2)
    c2 = Dense(1600, activation="relu")(c2)
    ysplit = []
    for j in range(len(ydims)):
        yi = Dense(ydims[j] ** 2, activation="relu")(c2)
        ysplit.append(yi)
    y1d = Reshape((1, 1, 1))(ysplit[0])
    return Model(inputs=inputs, outputs=[y1d])

for i in range(n_folds):
    tr_wrangler.set_fold(0)
    va_wrangler.set_fold(0)
    tr_wrangler.set_mode("train")
    va_wrangler.set_mode("val")

    model = build_model_1(layer_dims, 5)
    model.compile(loss="mse")
    print(model.summary())

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
    model.save_weights(save_dir + "/last_epoch_agb_only.h5")
    ### save executable

    print(callback1.logs['loss'])
    print(callback1.logs['time'])
    print(callback1.logs['val_accuracy'])
    print()