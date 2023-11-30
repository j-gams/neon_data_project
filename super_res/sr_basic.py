### sr_basic
import numpy as np

from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, Concatenate, Flatten, Reshape, MaxPooling2D

rootdir = "../data/sr_1"
n_layers = 5
n_folds = 1
layer_dims = [3,
              3,
              3,
              3,
              1]
batch_size = 32
buffer_nodata = -99
x_ids = [0, 1, 2, 3]
y_ids = [4]