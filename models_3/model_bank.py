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

### Contents...
### - model maker
### - losscallback
### - Cascade 1
### - Flat 1

### model importer
def make_model(modeltype):
    if "cascade1" in modeltype:
        return model_cascade1()
    elif "flat1" in modeltype:
        return model_flat1()

def save_metrics(metrics_dict, modeldir, fold):
    #self.base_model.save_weights(self.modeldir + "/model_" + self.name + ".h5")
    with open(modeldir + "/metric_" + str(fold) + ".txt", "wb") as metric_out:
        pickle.dump(metrics_dict, metric_out)

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
        self.logs["time"].append(ctime)
        self.lasttime = ctime
        for k in keys:
            self.logs[k].append(logs.get(k))

    def resume_from_load(self, loaded_logs):
        self.init = True
        self.logs = loaded_logs

### MODEL: Cascade 1
class model_cascade1:
    def __init__(self):
        self.base_model = None

    def make_base_model(self, ldims, yoff, uniquedimssorted, freq, ydims_unique, ydims_freq):
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

        convs2 = [[] for ii in range(len(uniquedimssorted))]
        for i in range(len(convs1)):
            if i == 0:
                ### 1 -> 1
                convs2[0].append(convs1[0][0])
            else:
                c2 = Concatenate(axis=3)(convs1[i])
                if i == 1:
                    ### 2 -> 2
                    c2 = Conv2D(filters=8 * freq[1], kernel_size=(1, 1), strides=(1, 1))(c2)
                if i == 2:
                    ### 17 -> 8
                    c2 = Conv2D(filters=8 * 2 * freq[2], kernel_size=(3, 3), strides=(2, 2), padding="valid")(c2)
                    ### 8 -> 2
                    c2 = Conv2D(filters=8 * 4 * freq[2], kernel_size=(5, 5), strides=(4, 4), padding="same")(c2)
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

    def setup(self, model_params, model_name):
        self.v = model_params["verbosity"]
        self.layer_dims = model_params["layerdims"]
        self.name = model_name
        self.modeldir = model_params["dir"]
        self.x_ids = model_params["x_layers"]
        self.y_ids = model_params["y_layers"]

        self.training_loss = model_params["training_loss"]
        self.monitor_loss = model_params["monitor_loss"]
        ### compute combinations
        unique_layer_dims = []
        unique_ydims = []
        # layer_cat = []

        ### TODO - make this not dumb
        for i in range(len(self.layer_dims)):
            if i in self.x_ids and self.layer_dims[i] not in unique_layer_dims:
                unique_layer_dims.append(self.layer_dims[i])
            if i in self.y_ids and self.layer_dims[i] not in unique_ydims:
                unique_ydims.append(self.layer_dims[i])
        sort_unique = list(unique_layer_dims)
        sort_unique.sort()
        unique_freq = [0 for ii in range(len(sort_unique))]
        unique_ydims.sort()
        unique_yfreq = [0 for ii in range(len(unique_ydims))]
        for i in range(len(self.layer_dims)):
            if i in self.x_ids:
                for j in range(len(sort_unique)):
                    if self.layer_dims[i] == sort_unique[j]:
                        unique_freq[j] += 1
            else:
                for j in range(len(unique_ydims)):
                    if self.layer_dims[i] == unique_ydims[j]:
                        unique_yfreq[j] += 1

        self.base_model = self.make_base_model(self.layer_dims, self.y_ids[0], unique_layer_dims,
                                               unique_freq, unique_ydims, unique_yfreq)
        self.base_model.compile(loss=self.training_loss)
        if self.v > 0:
            print(self.base_model.summary())
        callback1 = lossCallback()
        callback2 = ModelCheckpoint(self.modeldir + "/chkpt_" + self.name + ".h5",
                                    monitor = "val_mean_squared_error", verbose=2, mode="min",
                                    save_best_only=True, save_freq="epoch", save_weights_only=True)
        self.callbacks = [callback1, callback2]

    def load(self):
        self.base_model.load_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "rb") as cblog:
            picklelog = pickle.load(cblog)
        self.callbacks[0].resume_from_load(self, picklelog)

    def fit(self, train_data, val_data, n_epochs):
        self.base_model.fit(train_data, callbacks=self.callbacks, epochs=n_epochs, validation_data=val_data,
                            verbose=self.v)

    def predict(self, val_data):
        ### iterate over y layers
        yhats = [[] for iii in range(len(self.y_ids))]
        ys = [[] for iii in range(len(self.y_ids))]
        for i in range(len(val_data)):
            ### val y at batch i
            ys_i = val_data[i][1]
            y_hats_i = self.base_model(val_data[i][0])
            for j in range(len(self.y_ids)):
                for elty in ys_i[j]:
                    ys[j].append(elty)
                for elth in y_hats_i[j]:
                    yhats[j].append(elth)

        for i in range(len(self.y_ids)):
            yhats[i] = np.array(yhats[i])
            ys[i] = np.array(ys[i])

        self.recent_ys = ys
        self.recent_yhats = yhats
        return ys, yhats

    def save(self):
        self.base_model.save_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "wb") as cblog:
            pickle.dump(self.callbacks[0].logs, cblog)

### MODEL: Flat 1
class model_flat1:
    def __init__(self):
        self.base_model = None

    def make_base_model(self, ldims, yoff, uniquedimssorted, freq, ydims_unique, ydims_freq):
        xdims = ldims[:yoff]
        ydims = ldims[yoff:]
        inputs = []
        convs = []
        for i in range(len(xdims)):
            inputs.append(Input(shape=(xdims[i], xdims[i], 1)))
            if xdims[i] < 3:
                c1 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1))(inputs[i])
                # c1 = Conv2D(filters=16, kernel_size=(1,1), strides=(1,1))(c1)
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

    def setup(self, model_params, model_name):
        self.v = model_params["verbosity"]
        self.layer_dims = model_params["layerdims"]
        self.name = model_name
        self.modeldir = model_params["dir"]
        self.x_ids = model_params["x_layers"]
        self.y_ids = model_params["y_layers"]

        self.training_loss = model_params["training_loss"]
        self.monitor_loss = model_params["monitor_loss"]
        ### compute combinations
        unique_layer_dims = []
        unique_ydims = []
        # layer_cat = []

        ### TODO - make this not dumb
        for i in range(len(self.layer_dims)):
            if i in self.x_ids and self.layer_dims[i] not in unique_layer_dims:
                unique_layer_dims.append(self.layer_dims[i])
            if i in self.y_ids and self.layer_dims[i] not in unique_ydims:
                unique_ydims.append(self.layer_dims[i])
        sort_unique = list(unique_layer_dims)
        sort_unique.sort()
        unique_freq = [0 for ii in range(len(sort_unique))]
        unique_ydims.sort()
        unique_yfreq = [0 for ii in range(len(unique_ydims))]
        for i in range(len(self.layer_dims)):
            if i in self.x_ids:
                for j in range(len(sort_unique)):
                    if self.layer_dims[i] == sort_unique[j]:
                        unique_freq[j] += 1
            else:
                for j in range(len(unique_ydims)):
                    if self.layer_dims[i] == unique_ydims[j]:
                        unique_yfreq[j] += 1

        self.base_model = self.make_base_model(self.layer_dims, self.y_ids[0], unique_layer_dims,
                                               unique_freq, unique_ydims, unique_yfreq)
        self.base_model.compile(loss=self.training_loss)
        if self.v > 0:
            print(self.base_model.summary())
        callback1 = lossCallback()
        callback2 = ModelCheckpoint(self.modeldir + "/chkpt_" + self.name + ".h5",
                                    monitor = "val_mean_squared_error", verbose=2, mode="min",
                                    save_best_only=True, save_freq="epoch", save_weights_only=True)
        self.callbacks = [callback1]#, callback2]
        print("done setting up")

    def load(self):
        self.base_model.load_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "rb") as cblog:
            picklelog = pickle.load(cblog)
        self.callbacks[0].resume_from_load(self, picklelog)

    def fit(self, train_data, val_data, n_epochs):
        print("fitting...")
        self.base_model.fit(train_data, callbacks=self.callbacks, epochs=n_epochs, validation_data=val_data,
                            verbose=2)

    def predict(self, val_data):
        ### iterate over y layers
        yhats = [[] for iii in range(len(self.y_ids))]
        ys = [[] for iii in range(len(self.y_ids))]
        for i in range(len(val_data)):
            ### val y at batch i
            ys_i = val_data[i][1]
            y_hats_i = self.base_model(val_data[i][0])
            for j in range(len(self.y_ids)):
                for elty in ys_i[j]:
                    ys[j].append(elty)
                for elth in y_hats_i[j]:
                    yhats[j].append(elth)

        for i in range(len(self.y_ids)):
            yhats[i] = np.array(yhats[i])[:,:,:,0]
            ys[i] = np.array(ys[i])
            print("shape sanity check", i, "-", yhats[i].shape, ys[i].shape)
        self.recent_ys = ys
        self.recent_yhats = yhats
        return ys, yhats

    def save(self):
        self.base_model.save_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "wb") as cblog:
            pickle.dump(self.callbacks[0].logs, cblog)

