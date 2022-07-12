### TEST MODEL 1

### TEMPLATE???
### imports
import numpy as np
from tensorflow import keras
import tensorflow.keras.layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

class test_conv:
    def __init__ (self, modelname, saveloc, imgsize, save_checks, verbosity=2):
        self.verbosity = verbosity

        self.modelname = modelname
        self.imgsize = imgsize
        self.save_checks = save_checks
        self.model = keras.models.Sequential([keras.layers.InputLayer(input_shape=imgsize),
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
        if verbosity >= 2:
            print(self.model.summary())
        if self.save_checks:
            checkpoint_callbk = tf.keras.callbacks.ModelCheckpoint(
                    modelname + ".h5",
                    monitor=

    def train(self):

    def predict(self):
