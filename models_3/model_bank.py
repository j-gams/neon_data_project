import numpy as np

from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, Concatenate, Flatten, Reshape, MaxPooling2D, Conv2DTranspose

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
    elif "cascade2" in modeltype:
        return model_cascade2()
    elif "flat1" in modeltype:
        return model_flat1()
    elif "flat2" in modeltype:
        return model_flat2()

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

        ###

        convs2 = [[] for ii in range(len(uniquedimssorted) - 1)]
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

        ###

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
        yfinalnames = ["agb", "wue", "esi"]
        ncounter = 0
        for j in range(len(ydims_unique)):
            if ydims_freq[j] == 1:
                yj2 = Reshape((ydims_unique[j], ydims_unique[j], 1), name=yfinalnames[ncounter])(ysplit[j])
                ysplit2.append(yj2)
                associated_dim.append(ydims_unique[j])
                ncounter += 1
            else:
                for i in range(ydims_freq[j]):
                    yj2a = Dense(ydims[j] ** 2, activation="relu")(ysplit[j])
                    yj2b = Reshape((ydims_unique[j], ydims_unique[j], 1), name=yfinalnames[ncounter])(yj2a)
                    ysplit2.append(yj2b)
                    associated_dim.append(ydims_unique[j])
                    ncounter += 1
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

        print(self.x_ids, self.y_ids)
        ### TODO - make this not dumb
        # layer_cat = []
        for i in range(len(self.layer_dims)):
            if i in self.x_ids and self.layer_dims[i] not in unique_layer_dims:
                unique_layer_dims.append(self.layer_dims[i])
            if i in self.y_ids and self.layer_dims[i] not in unique_ydims:
                unique_ydims.append(self.layer_dims[i])

        # for i in range(len(unique_layer_dims)):
        #    for j in range(len(layer_dims)):
        #        if layer_dims[j] == unique_layer_dims[i]:
        #            layer_cat.append(i)

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

        print(unique_layer_dims, unique_freq)
        print(unique_ydims, unique_yfreq)

        self.base_model = self.make_base_model(self.layer_dims, self.y_ids[0], sort_unique,
                                               unique_freq, unique_ydims, unique_yfreq)
        self.base_model.compile(loss=self.training_loss)
        if self.v > 0:
            print(self.base_model.summary())
        callback1 = lossCallback()
        callback2 = ModelCheckpoint(self.modeldir + "/chkpt_" + self.name + ".h5",
                                    monitor = "val_mean_squared_error", verbose=2, mode="min",
                                    save_best_only=True, save_freq="epoch", save_weights_only=True)
        self.callbacks = [callback1]#, callback2]

    def load(self):
        self.base_model.load_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "rb") as cblog:
            picklelog = pickle.load(cblog)
        self.callbacks[0].resume_from_load(picklelog)

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
            yhats[i] = np.array(yhats[i])[:, :, :, 0]
            ys[i] = np.array(ys[i])
            print("shape sanity check", i, "-", yhats[i].shape, ys[i].shape)

        self.recent_ys = ys
        self.recent_yhats = yhats
        return ys, yhats

    def save(self):
        self.base_model.save_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "wb") as cblog:
            pickle.dump(self.callbacks[0].logs, cblog)

class TensorAddLayer(keras.layers.Layer):
    def __init__(self, adims, bdims):
        super().__init__()
        self.factor = math.ceil(adims / bdims)
        self.adims = adims
        self.bdims = bdims

    def call(self, inputs):
        tiled_tensor = keras.backend.tile(inputs[1], [1, self.factor, self.factor, 1])
        return tiled_tensor + inputs[0]

### MODEL: Cascade 2 - new basic cascade
class model_cascade2:
    def __init__(self):
        self.base_model = None

    def make_base_model(self, ldims, yoff, uniquedimssorted, freq, ydims_unique, ydims_freq, variant="a",
                        singletask=False):
        xdims = ldims[:yoff]
        ydims = ldims[yoff:]
        inputs = []
        convs1 = [[] for ii in range(len(uniquedimssorted))]
        convs2 = []
        if variant == "e":
            evariant_tofreq = max(list(freq))

        print("building variant", variant)
        print(uniquedimssorted)
        print(freq)

        convdims = list(uniquedimssorted)
        sfreq = list(freq)
        ### step 1 - gather layers into groups
        for i in range(len(xdims)):
            inputs.append(Input(shape=(xdims[i], xdims[i], 1)))
            for j in range(len(uniquedimssorted)):
                if xdims[i] == uniquedimssorted[j]:
                    convs1[j].append(inputs[i])
                    break

        ### step 2 - concatenate
        for j in range(len(uniquedimssorted)):
            convs2.append(Concatenate(axis=3)(convs1[j]))

        ### step 3 - action
        for j in range(len(uniquedimssorted)):
            if variant == "a":
                if uniquedimssorted[j] <= 2:
                    ### 1 -> 1 or 2 -> 2
                    convs2[j] = Conv2D(filters=8*freq[j], kernel_size=(1, 1), strides=(1, 1),
                                       padding="same")(convs2[j])
                else:
                    ### 1st conv
                    ### 34 -> 32
                    ### or 27 -> 25
                    convs2[j] = Conv2D(filters=8*freq[j], kernel_size=(3, 3), strides=(1, 1),
                                       padding="valid")(convs2[j])
                    convdims[j] -= 2
            elif variant == "b":
                if uniquedimssorted[j] == 1:
                    ### 1 -> 2
                    convs2[j] = Conv2DTranspose(filters=8*freq[j], kernel_size=(2, 2),
                                                strides=(1, 1))(convs2[j])
                    convdims[j] = 2
                elif uniquedimssorted[j] == 2:
                    ### 2-> 2
                    convs2[j] = Conv2D(filters=8 * freq[j], kernel_size=(1, 1), strides=(1, 1),
                                       padding="same")(convs2[j])
                else:
                    ### 1st conv
                    ### 34 -> 32
                    ### or 27 -> 25
                    convs2[j] = Conv2D(filters=8 * freq[j], kernel_size=(3, 3), strides=(1, 1),
                                       padding="valid")(convs2[j])
                    convdims[j] -= 2
            elif variant == "c":
                if uniquedimssorted[j] <= 2:
                    ### 1 -> 1 or 2 -> 2
                    convs2[j] = Conv2D(filters=8*freq[j], kernel_size=(1, 1), strides=(1, 1),
                                       padding="same")(convs2[j])
                else:
                    ### 1st conv
                    ### 34 -> 32
                    ### or 27 -> 25
                    convs2[j] = Conv2D(filters=8*freq[j], kernel_size=(3, 3), strides=(1, 1),
                                       padding="valid")(convs2[j])
                    convdims[j] -= 2
            elif variant == "d":
                if uniquedimssorted[j] <= 2:
                    ### 1 -> 1 or 2 -> 2
                    convs2[j] = Conv2D(filters=8 * freq[j], kernel_size=(1, 1), strides=(1, 1),
                                       padding="same")(convs2[j])
                else:
                    ### 1st conv
                    ### 34 -> 32
                    ### or 27 -> 25
                    convs2[j] = Conv2D(filters=8 * freq[j], kernel_size=(3, 3), strides=(1, 1),
                                       padding="valid")(convs2[j])
                    convdims[j] -= 2
            elif variant == "e":
                #if uniquedimssorted[j] == 1:
                #    ### 1 -> 1 or 2 -> 2
                #    convs2[j] = Conv2D(filters=2 * evariant_tofreq, kernel_size=(1, 1), strides=(1, 1),
                #                       padding="same")(convs2[j])
                if uniquedimssorted[j] <= 2:
                    ### 2 -> 1
                    convs2[j] = Conv2D(filters=2 * evariant_tofreq, kernel_size=(1, 1), strides=(1, 1),
                                       padding="same")(convs2[j])
                    #convdims[j] = 1
                else:
                    ### 1st conv
                    ### 34 -> 32
                    ### or 27 -> 25
                    convs2[j] = Conv2D(filters=2 * evariant_tofreq, kernel_size=(3, 3), strides=(1, 1),
                                       padding="valid")(convs2[j])
                    convdims[j] -= 2

        if variant == "e":
            ### do special layer...
            evariant_workdim = max(convdims)
            etemp = [convs2[2]]
            if len(sfreq) > 1:
                ## adim bdim [inputa, inputb]
                etemp.append(TensorAddLayer(32, 1)([convs2[2], convs2[0]]))
                etemp.append(TensorAddLayer(32, 2)([convs2[2], convs2[1]]))
            else:
                print("uhoh")
            sfreq = [sum(sfreq)]
            convdims = [convdims[2]]
            convs2[0] = Concatenate(axis=3)(etemp)

        ### step 4 - action:
        for j in range(len(sfreq)):
            if variant == "a":
                if convdims[j] <= 2:
                    pass
                else:
                    ### 1st maxpool
                    ### 32 -> 16
                    ### or 25 -> 13
                    convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j]/2)
            elif variant == "b":
                if convdims[j] <= 2:
                    #convs2[j] = Concatenate(axis=3)(convs2[j])
                    pass
                else:
                    ### 1st maxpool
                    ### 32 -> 16
                    ### or 25 -> 13
                    convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "c":
                if convdims[j] <= 2:
                    #convs2[j] = Concatenate(axis=3)(convs2[j])
                    pass
                else:
                    ### 1st maxpool
                    ### 32 -> 16
                    ### or 25 -> 13
                    convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "d":
                if convdims[j] > 2:
                    ### 1st maxpool
                    ### 32 -> 16
                    ### or 25 -> 13
                    convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "e":
                ## 32 -> 32 or 27 -> 27
                convs2[j] = Conv2D(filters=8 * sfreq[j], kernel_size=(3, 3), strides=(1, 1),
                                       padding="same")(convs2[j])

        ### step 5 - action:
        for j in range(len(sfreq)):
            if variant == "a":
                if convdims[j] <= 2:
                    pass
                else:
                    ### 2nd conv
                    ### 16 -> 8 and 13 -> 7
                    convs2[j] = Conv2D(filters=16 * freq[j], kernel_size=(3, 3), strides=(2, 2),
                                       padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "b":
                if convdims[j] > 2:
                    ### 2nd conv
                    ### 16 -> 8 and 13 -> 7
                    convs2[j] = Conv2D(filters=16 * freq[j], kernel_size=(3, 3), strides=(2, 2),
                                       padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "c":
                if convdims[j] > 2:
                    ### 1st conv
                    ### 16 -> 14
                    ### or 13 -> 11
                    convs2[j] = Conv2D(filters=8 * freq[j], kernel_size=(3, 3), strides=(1, 1),
                                       padding="valid")(convs2[j])
                    convdims[j] -= 2
            elif variant == "d":
                if convdims[j] > 2:
                    ### 2nd conv
                    ### 16 -> 6 and 13 -> 5
                    convs2[j] = Conv2D(filters=16 * freq[j], kernel_size=(3, 3), strides=(3, 3),
                                       padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 3)
            elif variant == "e":
                ### 1st maxpool
                ### 32 -> 16
                ### or 25 -> 13
                convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                convdims[j] = math.ceil(convdims[j] / 2)

        ### step 6 - action:
        for j in range(len(sfreq)):
            if variant == "a":
                if convdims[j] <= 2:
                    pass
                else:
                    ### 2nd maxpool
                    ### 8 -> 4 and 7 -> 4
                    convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "b":
                if convdims[j] > 2:
                    ### 2nd maxpool
                    ### 8 -> 4 and 7 -> 4
                    convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "c":
                if convdims[j] > 2:
                    ### 2nd maxpool
                    ### 14 -> 7 and 11 -> 6
                    convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "d":
                if convdims[j] > 2:
                    ### 2nd conv
                    ### 6 -> 2 and 5 -> 2
                    convs2[j] = Conv2D(filters=16 * freq[j], kernel_size=(3, 3), strides=(3, 3),
                                       padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 3)
            elif variant == "e":
                ### 2nd conv
                ### 16 -> 6 and 13 -> 5
                convs2[j] = Conv2D(filters=16 * sfreq[j], kernel_size=(3, 3), strides=(3, 3),
                                   padding="same")(convs2[j])
                convdims[j] = math.ceil(convdims[j] / 3)

        ### step 7 - action:
        for j in range(len(sfreq)):
            if variant == "a":
                if convdims[j] <= 2:
                    pass
                else:
                    ### 3rd conv
                    ### 4 -> 2 and 4 -> 2
                    convs2[j] = Conv2D(filters=16 * freq[j], kernel_size=(3, 3), strides=(2, 2),
                                       padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "b":
                if convdims[j] > 2:
                    ### 3rd conv
                    ### 4 -> 2 and 4 -> 2
                    convs2[j] = Conv2D(filters=16 * freq[j], kernel_size=(3, 3), strides=(2, 2),
                                       padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "c":
                ### 3rd conv
                if convdims[j] > 2:
                    ### 7 -> 5
                    ### or 6 -> 4
                    convs2[j] = Conv2D(filters=8 * freq[j], kernel_size=(3, 3), strides=(1, 1),
                                       padding="valid")(convs2[j])
                    convdims[j] -= 2
            elif variant == "e":
                ### 2nd conv
                ### 6 -> 2 and 5 -> 2
                convs2[j] = Conv2D(filters=16 * freq[j], kernel_size=(3, 3), strides=(3, 3),
                                   padding="same")(convs2[j])
                convdims[j] = math.ceil(convdims[j] / 3)

        ### step 7.5 - action:
        for j in range(len(sfreq)):
            if variant == "c":
                if convdims[j] > 2:
                    ### 3rd maxpool
                    ### 5 ->2 and 4->2
                    convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(convs2[j])
                    convdims[j] = 2

        print(convdims)
        run_concat = False
        if variant == "a":
            if len(convdims) > 1:
                convdims = [1, 2]
                convs2[1] = [convs2[1], convs2[2]]
                sfreq[1] = freq[1] + freq[2]
                run_concat = True
        if variant == "c":
            if len(convdims) > 1:
                convdims = [1, 2]
                convs2[1] = [convs2[1], convs2[2]]
                sfreq[1] = freq[1] + freq[2]
                run_concat = True
        if variant == "d":
            if len(convdims) > 1:
                convdims = [1, 2]
                convs2[1] = [convs2[1], convs2[2]]
                sfreq[1] = freq[1] + freq[2]
                run_concat = True
        ### e is already concatted

        ### step 8 - concat:
        for j in range(len(convdims)):
            if variant == "a" or variant == "c" or variant == "d":
                if convdims[j] == 2 and run_concat:
                    convs2[j] = Concatenate(axis=3)(convs2[j])
                else:
                    print(j, convdims[j])

        ### step 9 - conv
        for j in range(len(convdims)):
            if variant == "a":
                if convdims[j] == 2:
                    convs2[j] = Conv2D(filters=16 * sfreq[j], kernel_size=(2, 2), strides=(2, 2),
                           padding="same")(convs2[j])
            if variant == "c":
                if convdims[j] == 2:
                    convs2[j] = Conv2D(filters=16 * sfreq[j], kernel_size=(2, 2), strides=(2, 2),
                           padding="same")(convs2[j])
            if variant == "d":
                if convdims[j] == 2:
                    convs2[j] = Conv2D(filters=16 * sfreq[j], kernel_size=(2, 2), strides=(2, 2),
                           padding="same")(convs2[j])
                    convdims[j] = 1

        run_concat = False
        if variant == "a":
            if len(convdims) > 1:
                convdims = [1]
                convs2[0] = [convs2[0], convs2[1]]
                sfreq[0] += sfreq[0]
                run_concat = True
        if variant == "b":
            if len(convdims) > 1:
                convdims = [2, 2, 2]
                convs2[0] = [convs2[0], convs2[1], convs2[2]]
                sfreq[0] = freq[0] + freq[1] + freq[2]
                convs2[0] = Concatenate(axis=3)(convs2[0])
            else:
                print("nodims")
        if variant == "c":
            if len(convdims) > 1:
                convdims = [1]
                convs2[0] = [convs2[0], convs2[1]]
                sfreq[0] += sfreq[1]
                run_concat = True
        if variant == "d":
            if len(convdims) > 1:
                convdims = [1]
                convs2[0] = [convs2[0], convs2[1]]
                sfreq[0] += sfreq[1]
                run_concat = True
            print(convdims)

        ### step 10 - concat:
        for j in range(len(convdims)):
            if variant == "a" or variant == "c" or variant == "d":
                if convdims[j] == 1 and len(convdims) > 1:
                    convs2[j] = Concatenate(axis=3)(convs2[j])
                else:
                    print("oh no", j, convdims[j])

        ### step 12 - conv and flatten
        convs2[0] = Conv2D(filters=16 * sfreq[0], kernel_size=(1, 1), strides=(1, 1),
                           padding="same")(convs2[0])
        convres = Flatten()(convs2[0])
        #convres = Concatenate()(convs2)

        fc = Dense(1600, activation="relu")(convres)
        fc = Dense(1800, activation="relu")(fc)
        fc = Dense(1600, activation="relu")(fc)
        ysplit = []
        if singletask is None:
            for j in range(len(ydims_unique)):
                yj = Dense((ydims_unique[j] ** 2) * ydims_freq[j], activation="relu")(fc)
                ysplit.append(yj)
        else:
            ysplit = Dense((ydims[singletask] ** 2), activation="relu")(fc)
        associated_dim = []
        ysplit2 = []
        yfinalnames = ["agb", "wue", "esi"]
        ncounter = 0
        combineout = True
        if singletask is None:
            if combineout:
                for j in range(len(ydims_unique)):
                    if ydims_freq[j] == 1:
                        yj2 = Dense((ydims_unique[j] ** 2), activation="relu")(ysplit[j])
                        yj2 = Reshape((ydims_unique[j], ydims_unique[j], 1), name=yfinalnames[ncounter])(yj2)
                        ysplit2.append(yj2)
                        associated_dim.append(ydims_unique[j])
                        ncounter += 1
                    else:
                        yj2i = Concatenate()(ysplit)
                        yj2i = Dense(ydims[0] ** 2 + ydims[1] ** 2, activation="relu")(yj2i)
                        for i in range(ydims_freq[j]):
                            yj2a = Dense(ydims[j] ** 2, activation="relu")(yj2i)
                            yj2b = Reshape((ydims_unique[j], ydims_unique[j], 1), name=yfinalnames[ncounter])(yj2a)
                            ysplit2.append(yj2b)
                            associated_dim.append(ydims_unique[j])
                            ncounter += 1
            else:
                for j in range(len(ydims_unique)):
                    if ydims_freq[j] == 1:
                        yj2 = Reshape((ydims_unique[j], ydims_unique[j], 1), name=yfinalnames[ncounter])(ysplit[j])
                        ysplit2.append(yj2)
                        associated_dim.append(ydims_unique[j])
                        ncounter += 1
                    else:
                        for i in range(ydims_freq[j]):
                            yj2a = Dense(ydims[j] ** 2, activation="relu")(ysplit[j])
                            yj2b = Reshape((ydims_unique[j], ydims_unique[j], 1), name=yfinalnames[ncounter])(yj2a)
                            ysplit2.append(yj2b)
                            associated_dim.append(ydims_unique[j])
                            ncounter += 1
        else:
            ysplit2 = Reshape((ydims[singletask], ydims[singletask], 1), name=yfinalnames[singletask])(ysplit)
            associated_dim = ydims[singletask]
            ncounter += 1
        yfinal = []
        if singletask is None:
            for i in range(len(ydims)):
                ### find one w/ appropriate exit dim
                for j in range(len(ysplit2)):
                    if associated_dim[j] == ydims[i]:
                        yfinal.append(ysplit2.pop(j))
                        associated_dim.pop(j)
                        break
        else:
            yfinal = ysplit2

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
        self.vari = model_params["variant"]
        self.singletask = model_params["singletask"]
        ### compute combinations
        unique_layer_dims = []
        unique_ydims = []
        # layer_cat = []

        print(self.x_ids, self.y_ids)
        ### TODO - make this not dumb
        # layer_cat = []
        for i in range(len(self.layer_dims)):
            if i in self.x_ids and self.layer_dims[i] not in unique_layer_dims:
                unique_layer_dims.append(self.layer_dims[i])
            if i in self.y_ids and self.layer_dims[i] not in unique_ydims:
                unique_ydims.append(self.layer_dims[i])

        # for i in range(len(unique_layer_dims)):
        #    for j in range(len(layer_dims)):
        #        if layer_dims[j] == unique_layer_dims[i]:
        #            layer_cat.append(i)

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

        print(unique_layer_dims, unique_freq)
        print(unique_ydims, unique_yfreq)

        self.base_model = self.make_base_model(self.layer_dims, self.y_ids[0], sort_unique,
                                               unique_freq, unique_ydims, unique_yfreq,
                                               variant=self.vari, singletask=self.singletask)
        self.base_model.compile(loss=self.training_loss)
        if self.v > 0:
            print(self.base_model.summary())
        callback1 = lossCallback()
        callback2 = ModelCheckpoint(self.modeldir + "/chkpt_" + self.name + ".h5",
                                    monitor = "val_mean_squared_error", verbose=2, mode="min",
                                    save_best_only=True, save_freq="epoch", save_weights_only=True)
        self.callbacks = [callback1]#, callback2]

    def load(self):
        self.base_model.load_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "rb") as cblog:
            picklelog = pickle.load(cblog)
        self.callbacks[0].resume_from_load(picklelog)

    def fit(self, train_data, val_data, n_epochs):
        if self.singletask is not None:
            train_data.set_single_y(self.singletask)
            val_data.set_single_y(self.singletask)
        self.base_model.fit(train_data, callbacks=self.callbacks, epochs=n_epochs, validation_data=val_data,
                            verbose=self.v)
        if self.singletask is not None:
            train_data.set_multi_y()
            val_data.set_multi_y()

    def predict(self, val_data):
        if self.singletask is not None:
            val_data.set_single_y(self.singletask)
        yhats = [[] for iii in range(len(val_data.use_y_ids))]
        ys = [[] for iii in range(len(val_data.use_y_ids))]
        ### iterate over batches?
        for i in range(len(val_data)):
            ### val y at batch i
            ys_i = val_data[i][1]
            y_hats_i = self.base_model(val_data[i][0])
            if self.singletask is not None:
                y_hats_i = [y_hats_i]
                #print(y_hats_i.shape)
            ### iterate over y layers
            for j in range(len(val_data.use_y_ids)):
                for elty in ys_i[j]:
                    ys[j].append(elty)
                for elth in y_hats_i[j]:
                    yhats[j].append(elth)

        for i in range(len(val_data.use_y_ids)):
            yhats[i] = np.array(yhats[i])[:, :, :, 0]
            ys[i] = np.array(ys[i])
            print("shape sanity check", i, "-", yhats[i].shape, ys[i].shape)

        if self.singletask is not None:
            val_data.set_multi_y()

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
        print("loading model...")
        self.base_model.load_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "rb") as cblog:
            picklelog = pickle.load(cblog)
        self.callbacks[0].resume_from_load(picklelog)

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

### MODEL: Flat 2
class model_flat2:
    def __init__(self):
        self.base_model = None

    def make_base_model(self, ldims, yoff, uniquedimssorted, freq, ydims_unique, ydims_freq):
        xdims = ldims[:yoff]
        ydims = ldims[yoff:]
        inputs = []
        convs = []
        for i in range(len(xdims)):
            """inputs.append(Input(shape=(xdims[i], xdims[i], 1)))
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
            convs.append(Flatten()(c1))"""
            inputs.append(Input(shape=(xdims[i], xdims[i], 1)))
        c1 = Concatenate(axis=3)(inputs)
        ### 34 -> 32 or 27 -> 25
        c1 = Conv2D(filters=40, kernel_size=(3, 3), strides=(1, 1), padding="valid")(c1)
        ### 32 -> 30 or 25 -> 23
        c1 = Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding="valid")(c1)
        ### 30 -> 28 or 23 -> 21
        c1 = Conv2D(filters=157, kernel_size=(3, 3), strides=(1, 1), padding="valid")(c1)
        ### 30 -> 15 or 23 -> 12
        c1 = Conv2D(filters=304, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
        ### 15 -> 8 or 12 -> 6
        c1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(c1)

        ### 8 -> 4 or 6 -> 3
        c1 = Conv2D(filters=304, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
        ### 4 -> 2 or 3 -> 2
        c1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(c1)
        merge = Flatten()(c1)
        c2 = Dense(1600, activation="relu")(merge)
        c2 = Dense(1800, activation="relu")(c2)
        c2 = Dense(1600, activation="relu")(c2)

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
        self.callbacks[0].resume_from_load(picklelog)

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

