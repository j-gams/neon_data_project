### Framework for managing training models

### imports...
import numpy as np
import sys

sys.path.insert(0, '../create_data')

from dat_obj import datacube_loader
from datacube_set import satimg_set

verbosity = 2
def qprint(pstr, importance):
    if importance <= verbosity:
        print(pstr)

### TODO - read in file params from config file... eventually

### training parameters
dataset = "data_interpolated"
folding = "test_fold"
models = []
model_hparams = []
save_names = []


