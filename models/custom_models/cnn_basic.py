### Written by Jerry Gammie @j-gams

### TEST MODEL 1

### TEMPLATE???
### imports
import numpy as np
from tensorflow import keras
import tensorflow.keras.layers
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


class cnn:
    def __init__(self, hparam_dict, save_dir):
        # model_name, save_location, input_size, save_checkpoints, train_metric, verbosity=2):
        init_count = 0
        self.verbosity = 0
        self.reload_best = True
        self.save_last = True

        self.crdict = dict()
        self.dropmode = "none"
        self.dropout = []
        self.add_noise = False
        self.noise_level = 0

        train_metric = "mean_squared_error"
        for key in hparam_dict:
            if key == "model_name":
                self.modelname = hparam_dict[key]
                init_count += 1
            elif key == "save_location":
                self.saveloc = hparam_dict[key]
                init_count += 1
            elif key == "input_size":
                self.imgsize = hparam_dict[key]
                init_count += 1
            elif key == "save_checkpoints":
                self.savechecks = hparam_dict[key]
                init_count += 1
            elif key == "train_metric":
                train_metric = hparam_dict[key]
                init_count += 1
            elif key == "epochs":
                self.n_epochs = hparam_dict[key]
                init_count += 1
            elif key == "use_best":
                self.reload_best = hparam_dict[key]
            elif key == "save_last_epoch":
                self.save_last = hparam_dict[key]
            elif key == "verbosity":
                self.verbosity = hparam_dict[key]
            elif key == "dropout":
                self.dropmode = hparam_dict[key]["mode"]
                self.dropout = hparam_dict[key]["channels"]
            elif key == "optimizer":
                self.optimizerstr = hparam_dict[key]
            elif key == "arch":
                self.archdicts = hparam_dict[key]
            elif key == "noise":
                self.add_noise = True
                self.noise_level = hparam_dict[key]

        ### ARCHDICT STRUCTURE
        ### - "convlayers"
        ###   - [layerdict {lname:"", lparams:[], lquant}, ...]
        ### - "denselayers"

        if self.dropmode == "keep":
            self.keeplen = len(self.dropout)
        elif self.dropmode == "drop":
            self.keeplen = self.imgsize[2] - len(self.dropout)
        else:
            self.keeplen = self.imgsize[2]
        self.imgsize = list(self.imgsize)
        self.imgsize[2] = self.keeplen
        self.imgsize = tuple(self.imgsize)
        print("***IMGSIZE", self.imgsize)
        if init_count != 6:
            ### did not initialize ok
            print("model not initialized correctly!")

        self.metricset = ["mean_squared_error", "mean_absolute_error"]
        if train_metric not in self.metricset:
            self.tmetric = "mean_squared_error"
        else:
            self.tmetric = train_metric

        self.save_dir = save_dir

        archlist = [keras.layers.InputLayer(input_shape=self.imgsize)]
        for layerdict in self.archdicts:
            archlist.extend(self.convert_dict_layer(layerdict[0], layerdict[1], layerdict[2]))
        archlist.append(keras.layers.Dense(1))


        self.model = keras.models.Sequential(archlist)

        print(self.model.summary())
        self.model.compile(loss=self.tmetric, metrics=self.metricset,
                           optimizer=keras.optimizers.Adam(learning_rate=0.0001))
        self.callbacks = []
        if self.savechecks:
            callback = ModelCheckpoint(self.save_dir + "/checkpoint.h5",
                                       monitor="val_mean_squared_error",
                                       verbose=2,
                                       mode="min",
                                       save_best_only=True,
                                       save_freq="epoch",
                                       save_weights_only=True)
            self.callbacks.append(callback)

    def convert_initializer(self, paramlist):
        if paramlist[0] == "uniform":
            return keras.initializers.RandomUniform(minval=paramlist[1], maxval=paramlist[2], seed=paramlist[3])

    def convert_dict_layer(self, lname, params_list, quant):
        retlist = []

        if lname == "conv2d":
            for i in range(quant):
                retlist.append(keras.layers.Conv2D(filters=params_list[0], kernel_size=params_list[1],
                                                   strides=params_list[2], padding=params_list[3],
                                                   activation=params_list[4]))
        elif lname == "batchnorm":
            for i in range(quant):
                retlist.append(keras.layers.BatchNormalization())
        elif lname == "avgpooling2d":
            for i in range(quant):
                retlist.append(keras.layers.AveragePooling2D(params_list[0], params_list[1]))
        elif lname == "maxpooling2d":
            for i in range(quant):
                retlist.append(keras.layers.MaxPooling2D(params_list[0], params_list[1]))
        elif lname == "dense":
            for i in range(quant):
                if params_list[2] is not None:
                    retlist.append(keras.layers.Dense(params_list[0], activation=params_list[1],
                                                  kernel_initializer=self.convert_initializer(params_list[2])))
                else:
                    retlist.append(keras.layers.Dense(params_list[0], activation=params_list[1]))
        elif lname == "flatten":
            return [keras.layers.Flatten()]
        elif lname == "res":
            pass
        return retlist


    def train(self, train_data, validation_data):
        self.change_restore(train_data, "c", "train")
        self.change_restore(validation_data, "c", "val")
        og_noise = train_data.noise_mode
        og_nlevel = train_data.noise_level
        train_data.noise_mode = self.add_noise
        train_data.noise_level = self.noise_level
        self.model.fit(train_data, callbacks=self.callbacks, epochs=self.n_epochs, validation_data=validation_data,
                       verbose=2)
        if self.save_last:
            self.model.save_weights(self.save_dir + "/last_epoch.h5")
        if self.reload_best and self.savechecks:
            self.model.load_weights(self.save_dir + "/checkpoint.h5")
        self.change_restore(train_data, "r", "train")
        self.change_restore(validation_data, "r", "val")

    def predict(self, x_predict, typein="simg"):
        self.change_restore(x_predict, "c", "predict")
        if typein == "simg":
            dumb_out = []
            for i in range(len(x_predict)):
                dumb_out.append(self.model(x_predict[i][0]))
            ret_y = np.array(dumb_out).reshape(-1).flatten()
        self.change_restore(x_predict, "r", "predict")
        return ret_y

    def change_restore(self, data, c_r, name):
        if c_r == "c":
            self.crdict[name] = [data.flat_mode,
                                 data.keep_ids,
                                 data.drop_channels]
            data.set_flatten(False)
            if self.dropmode == "keep":
                data.set_keeps(self.dropout)
                data.set_drops(data.keeps_to_drops())
            elif self.dropmode == "drop":
                data.set_drops(self.dropout)
                keepsl = []
                for i in range(data.nchannels):
                    if i not in self.dropout:
                        keepsl.append(i)
                data.set_keeps(keepsl)
            else:
                data.set_drops([])
                data.set_keeps(data.drops_to_keeps())
        else:
            data.set_flatten(self.crdict[name][0])
            data.set_keeps(self.crdict[name][1])
            data.set_drops(self.crdict[name][2])

    def load_best(self):
        pass

    def load_last(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
