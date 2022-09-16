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

class test_conv:
    def __init__ (self, hparam_dict, save_dir):# model_name, save_location, input_size, save_checkpoints, train_metric, verbosity=2):
        init_count = 0
        self.verbosity = 0
        self.reload_best = True
        self.save_last = True

        self.crdict = dict()
        self.dropmode = "none"
        self.dropout = []
        
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

        #if self.optimizerstr == "adam":
        #    pass

        #self.modelname = model_name
        #self.imgsize = input_size
        #self.save_checks = save_checkpoints
        self.model = keras.models.Sequential([keras.layers.InputLayer(input_shape=self.imgsize),
                                         keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding='same',
                                             activation='relu'),
                                         keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding='same',
                                             activation='relu'),
                                         keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=2,
                                                                  padding='same',
                                                                  activation='relu'),
                                         keras.layers.Flatten(),
                                         keras.layers.Dense(1024, activation='relu', kernel_initializer=
                                            keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)),
                                         keras.layers.Dense(1024, activation='relu', kernel_initializer=
                                            keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)),
                                         keras.layers.Dense(64, activation='relu', kernel_initializer=
                                            keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)),
                                         keras.layers.Dense(1)])

        """self.model = keras.models.Sequential([keras.layers.InputLayer(input_shape=self.imgsize),
                                              keras.layers.Conv2D(filters=132, kernel_size=(3, 3), strides=2,
                                                                  padding='same',
                                                                  activation='relu'),
                                              keras.layers.MaxPooling2D(2, 2),
                                              keras.layers.Conv2D(filters=264, kernel_size=(3, 3), strides=2,
                                                                  padding='same',
                                                                  activation='relu'),
                                              keras.layers.Flatten(),
                                              keras.layers.Dense(512, activation='relu'),
                                              keras.layers.Dense(256, activation='relu'),
                                              keras.layers.Dense(32, activation='relu'),
                                              keras.layers.Dense(1)])"""
        if self.verbosity >= 2:
            print(self.model.summary())
        self.model.compile(loss=self.tmetric, metrics=self.metricset, optimizer=keras.optimizers.Adam(learning_rate=0.0001))
        self.callbacks = []
        if self.savechecks:
            callback = ModelCheckpoint(self.save_dir + "/checkpoint.h5",
                    monitor="val_mean_squared_error",
                    verbose=2,
                    mode="min",
                    save_best_only=True,
                    save_freq="epoch",
                    save_weights_only = True)
            self.callbacks.append(callback)

    def train(self, train_data, validation_data):
        self.change_restore(train_data, "c", "train")
        self.change_restore(validation_data, "c", "val")
        self.model.fit(train_data, callbacks=self.callbacks, epochs=self.n_epochs, validation_data=validation_data, verbose = 2)#self.verbosity)
        if self.save_last:
            self.model.save_weights(self.save_dir + "/last_epoch.h5")
        if self.reload_best and self.savechecks:
            self.model.load_weights(self.save_dir + "/checkpoint.h5")
        self.change_restore(train_data, "r", "train")
        self.change_restore(validation_data, "r", "val")

    def predict(self, x_predict, typein="simg"):
        #print(type(x_predict))
        self.change_restore(x_predict, "c", "predict")
        if typein == "simg":
            dumb_out = []
            #og_ret = x_predict.return_format
            #x_predict.set_return("x")
            for i in range(len(x_predict)):
                dumb_out.append(self.model(x_predict[i][0]))
            ret_y = np.array(dumb_out).reshape(-1).flatten()
            #self.model(x_predict)
            #x_predict.set_return(og_ret)
        self.change_restore(x_predict, "r", "predict")
        return ret_y

    def change_restore(self, data, c_r, name):
        if c_r == "c":
            self.crdict[name] = [data.flat_mode,
                                 data.keep_ids,
                                 data.drop_channels]
            #data.set_return("x")
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
           #data.set_return(self.crdict[name][0])
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
