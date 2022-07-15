### TEST MODEL 1

### TEMPLATE???
### imports
import numpy as np
from tensorflow import keras
import tensorflow.keras.layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

class test_conv:
    def __init__ (self, hparam_dict):# model_name, save_location, input_size, save_checkpoints, train_metric, verbosity=2):
        init_count = 0
        self.verbosity = 2
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
            elif key == "verbosity":
                self.verbosity = hparam_dict[key]
        
        if init_count != 6:
            ### did not initialize ok
            print("model not initialized correctly!")

        self.metricset = ["mean_squared_error", "mean_absolute_error"]
        if train_metric not in self.metricset:
            self.tmetric = "mean_squared_error"
        else:
            self.tmetric = train_metric

        #self.modelname = model_name
        #self.imgsize = input_size
        #self.save_checks = save_checkpoints
        self.model = keras.models.Sequential([keras.layers.InputLayer(input_shape=self.imgsize),
                                         keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding='same',
                                             activation='relu'),
                                         keras.layers.MaxPooling2D(2, 2),
                                         keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding='same',
                                             activation='relu'),
                                         keras.layers.Flatten(),
                                         keras.layers.Dense(512, activation='relu'),
                                         keras.layers.Dense(256, activation='relu'),
                                         keras.layers.Dense(128, activation='sigmoid'),
                                         keras.layers.Dense(1)])
        if self.verbosity >= 2:
            print(self.model.summary())
        self.model.compile(loss=self.tmetric, metrics=self.metricset)
        self.callbacks = []
        if self.savechecks:
            callback = tf.keras.callbacks.ModelCheckpoint(modelname + ".h5",
                    monitor="val_"+self.tmetric,
                    verbose=1,
                    mode="min",
                    save_best_only=True,
                    save_freq="epoch",
                    save_weights_only = True)
            self.callbacks.append(callback)

    def train(self, train_data, validation_data):
        self.model.fit(train_data, callbacks=self.callbacks, epochs=self.n_epochs, validation_data=validation_data)

    def predict(self, x_predict, typein="simg"):
        print(type(x_predict))
        if typein == "simg":
            dumb_out = []
            og_ret = x_predict.return_format
            x_predict.set_return("x")
            for i in range(len(x_predict)):
                dumb_out.append(self.model(x_predict[i]))
            ret_y = np.array(dumb_out).reshape(-1).flatten()
            #self.model(x_predict)
            x_predict.set_return(og_ret)
        return ret_y
