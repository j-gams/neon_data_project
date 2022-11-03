import numpy as np
from tensorflow.keras.layers import Conv2D, Concatenate, BatchNormalization, Add, ReLU
from tensorflow.keras.callbacks import Callback

def firemodule (x_in, expand_filters, ratio_3x3, ratio_squeeze):
    n_3x3 = int(expand_filters * ratio_3x3)
    n_1x1 = expand_filters - n_3x3
    squeeze_filters = int(expand_filters * ratio_squeeze)
    ### squeeze
    x_in = Conv2D(squeeze_filters, (1, 1), strides=1, padding='valid', activation='relu')(x_in)
    ### expand (1x1 filters)
    x_1x1 = Conv2D(n_1x1, (1, 1), strides=1, padding='same', activation='relu')(x_in)
    ### expand (3x3 filters)
    x_3x3 = Conv2D(n_3x3, (3, 3), strides=1, padding='same', activation='relu')(x_in)
    ### concat to return
    return Concatenate(axis=3)([x_1x1, x_3x3])

def fireresidual (x_in, expand_filters, ratio_3x3, ratio_squeeze, batch_normalize):
    n_3x3 = int(expand_filters * ratio_3x3)
    n_1x1 = expand_filters - n_3x3
    squeeze_filters = int(expand_filters * ratio_squeeze)
    ### squeeze
    x_sqz = Conv2D(squeeze_filters, (1, 1), strides=1, padding='valid', activation='relu')(x_in)
    if batch_normalize:
        x_sqz = BatchNormalization()(x_sqz)
    ### expand (1x1 filters)
    x_1x1 = Conv2D(n_1x1, (1, 1), strides=1, padding='same', activation='relu')(x_sqz)
    ### expand (3x3 filters)
    x_3x3 = Conv2D(n_3x3, (3, 3), strides=1, padding='same', activation='relu')(x_sqz)

    ### concat to return
    y = Concatenate(axis=3)([x_1x1, x_3x3])
    if batch_normalize:
        y = BatchNormalization()(y)
    x_in = Conv2D(kernel_size=1, filters=expand_filters, strides=1, padding="same")(x_in)
    y = Add()([x_in, y])
    y = ReLU()(y)
    if batch_normalize:
        y = BatchNormalization()(y)
    return y

class fire_metaupdate:

    def __init__(self, params):
        self.fire_number = 0
        self.base_e = params[0]
        self.pct_3x3 = params[1]
        self.sqz_ratio = params[2]
        self.update_freq = params[3]
        self.incr_e = params[4]

    def get_ei (self, incr=True):
        val = self.base_e + (self.incr_e * int(1 / self.update_freq))
        if incr == True:
            self.fire_number += 1
        return val

class statsCallback(Callback):
    def __init__ (self):
        self.logs = dict()
        self.init = False

    def on_epoch_end(self, epoch, logs={}):
        keys = list(logs.keys())
        if not self.init:
            for k in keys:
                self.logs[k] = []
            self.init = True
        for k in keys:
            self.logs[k].append(logs.get(k))
