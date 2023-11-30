### sr_basic
import numpy as np

from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, Concatenate, Flatten, Reshape, MaxPooling2D

