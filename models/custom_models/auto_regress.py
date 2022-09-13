### Written by Jerry Gammie @j-gams

### autoencode

import numpy as np
from tensorflow import keras
import tensorflow.keras.layers
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge

class test_auto:
    def __init__ (self, hparam_dict, save_dir):
        self.verbosity = 2
        self.reload_best = True
        self.save_last = True
        self.crdict = dict()
        self.dropmode = "none"
        self.dropout = []
        self.denselayers = [1024, 512, 256]
        self.regress_step = "lr"
        self.rmodel = None

        self.enc_size = 16

        for key in hparam_dict:
            if key == "model_name":
                self.modelname = hparam_dict[key]
            elif key == "save_location":
                self.saveloc = hparam_dict[key]
            elif key == "encoding_size":
                self.enc_size = hparam_dict[key]
            elif key == "denselayers":
                self.denselayers = hparam_dict[key]
            elif key == "rstep":
                self.regress_step = hparam_dict[key]
            elif key == "rstep_params":
                self.rms_params = hparam_dict[key]
            elif key == "input_size":
                self.imgsize = hparam_dict[key]
            elif key == "save_checkpoints":
                self.savechecks = hparam_dict[key]
            elif key == "train_metric":
                train_metric = hparam_dict[key]
            elif key == "epochs":
                self.n_epochs = hparam_dict[key]
            elif key == "use_best":
                self.reload_best = hparam_dict[key]
            elif key == "save_last_epoch":
                self.save_last = hparam_dict[key]
            elif key == "verbosity":
                self.verbosity = hparam_dict[key]
            elif key == "dropout":
                self.dropmode = hparam_dict[key]["mode"]
                self.dropout = hparam_dict[key]["channels"]

        if self.dropmode == "keep":
            self.keeplen = len(self.dropout)
        elif self.dropmode == "drop":
            self.keeplen = self.imgsize[2] - len(self.dropout)
        else:
            self.keeplen = self.imgsize[2]
        self.imgsize = list(self.imgsize)
        self.imgsize[2] = self.keeplen
        self.imgsize = tuple(self.imgsize)
        
        self.save_dir = save_dir

        imgprod = self.imgsize[0] * self.imgsize[1] * self.imgsize[2]

        self.input_in = keras.Input(shape=self.imgsize)
        x = keras.layers.Conv2D(filters = 256, kernel_size=(3,3), strides=2, padding='same')(self.input_in)
        x = keras.layers.MaxPooling2D(2, 2)(x)
        x = keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=2, padding='same')(x)
        x = keras.layers.MaxPooling2D(2, 2)(x)
        x = keras.layers.Flatten()(x)
        for dlayer in self.denselayers:
            x = keras.layers.Dense(dlayer, activation='relu')(x)
            
        #x = keras.layers.Dense(1024, activation='relu')(x)
        #x = keras.layers.Dense(512, activation='relu')(x)
        self.encoded = keras.layers.Dense(self.enc_size, activation='relu')(x)
        encoded_in = keras.Input(shape=(self.enc_size,))
        xo = encoded_in
        for dlayer in reversed(self.denselayers):
            xo = keras.layers.Dense(dlayer, activation='relu')(xo)
        #xo = keras.layers.Dense(512, activation='relu')(self.encoded)
        #xo = keras.layers.Dense(1024, activation='relu')(xo)
        
        xo = keras.layers.Dense(imgprod, activation='relu')(xo)

        #encoded_in = keras.Input(shape=(self.enc_size,))
        xo = keras.layers.Reshape(self.imgsize)(xo)
        xo = keras.layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=2, activation="relu", padding="same")(xo)
        xo = keras.layers.Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=1, activation="relu", padding="same")(xo)
        xo = keras.layers.Conv2D(self.imgsize[2], kernel_size=(3,3), activation='relu', strides=2, padding='same')(xo)

        #self.autoencoder = keras.Model(self.input_in, xo)
        self.encodermodule = keras.Model(self.input_in, self.encoded)
        self.decodermodule = keras.Model(encoded_in, xo)
        self.autoencoder = keras.Model(self.input_in, self.decodermodule(self.encodermodule(self.input_in)))
        #encoded_in = keras.Input(shape=(self.enc_size,))
        #dec_layers = self.autoencoder.layers[-1]
        #self.decodermodule = keras.Model(encoded_in, dec_layers(encoded_in))

        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        print(self.autoencoder.summary())
        self.callbacks = []
        if self.savechecks:
            callback = ModelCheckpoint(self.save_dir + "/checkpoint.h5",
                    monitor="val_binary_crossentropy",
                    verbose=1,
                    mode="min",
                    save_best_only=True,
                    save_freq="epoch",
                    save_weights_only = True)
            self.callbacks.append(callback)
        
        
    ### train from scratch
    def train(self, train_data, validation_data):
        ### stage 1... 
        self.change_restore(train_data, "c", "train_ae", "ae")
        self.change_restore(validation_data, "c", "val_ae", "ae")

        self.train_ae(train_data, validation_data)

        self.change_restore(train_data, "c", "train_reg", "r")
        self.change_restore(validation_data, "c", "val_reg", "r")

        self.train_reg(train_data)
        
        self.change_restore(train_data, "r", "train_ae", "ae")
        self.change_restore(validation_data, "r", "val_ae", "ae")

    def train_ae(self, train_data, validation_data):
        self.autoencoder.fit(train_data, callbacks=self.callbacks, epochs=self.n_epochs,
                validation_data=validation_data, 
                verbose = self.verbosity)
        if self.save_last:
            self.autoencoder.save_weights(self.save_dir + "/last_epoch.h5")
        #if self.reload_best and self.savechecks:
        #    self.autoencoder.load_weights(self.save_dir + "/checkpoint.h5")
        
    def train_reg(self, train_data):
        #train_munge = []
        train_munge = np.zeros((train_data.X.shape[0], self.enc_size))
        for i in range(len(train_data)):
            x_i = train_data[i]
            #for j in range(len(x_i)):
            #fulltrain[i*train_data.batch_size:min(len(fulltrain),
            #                                          (i+1)*train_data.batch_size),
            #              :] = self.dtransform(train_data[i][0])
            train_munge[i*train_data.batch_size:min(len(x_i) + i*train_data.batch_size, (i+1)*train_data.batch_size), :] = self.encodermodule.predict(x_i)
        train_mungenp = np.array(train_munge)
        print("train_munge", train_mungenp.shape)
        if self.regress_step == "linr":
            self.rmodel = LinearRegression().fit(train_mungenp, train_data.y)
        elif self.regress_step == "lasr":
            self.rmodel = Lasso(alpha=self.rms_params["alpha"]).fit(train_mungenp, train_data.y)
        elif self.regress_step == "kerr":
            self.rmodel = KernelRidge(alpha=self.rms_params["alpha"], kernel="rbf").fit(train_mungenp, train_data.y)

    def predict(self, x_predict):
        dumb_out = []
        self.change_restore(x_predict, "c", "eval", "r")
        for i in range(len(x_predict)):
            dumb_out.append(self.rmodel.predict(self.encodermodule.predict(x_predict[i])))
        ret_y = np.array(dumb_out).reshape(-1).flatten()
        self.change_restore(x_predict, "r", "eval", "r")
        return ret_y

    def change_restore(self, data, c_r, name, mode):
        if c_r == "c":
            ### autoencoder mode
            if mode == "ae":
                self.crdict[name] = [data.return_format,
                                     data.flat_mode,
                                     data.keep_ids,
                                     data.drop_channels]
                data.set_return("xx")
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
            ### regression mode
            elif mode == "r":
                self.crdict[name] = [data.return_format,
                                     data.flat_mode,
                                     data.keep_ids,
                                     data.drop_channels]
                data.set_return("x")
                data.set_flatten(False)
                data.unshuffle()
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
           data.set_return(self.crdict[name][0])
           data.set_flatten(self.crdict[name][1])
           data.set_keeps(self.crdict[name][2])
           data.set_drops(self.crdict[name][3])
        
