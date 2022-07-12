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

### dataset parameters
dataset = "data_interpolated"
folding = "test_fold"

d_batch = 12
d_shuffle = [True, True, True]
d_batch = [d_batch, d_batch, d_batch]
d_meanstd = ["default", "default", "default"]
### TODO - move these to meta file
d_xref = 1
d_yref = 2
d_mmode = [True, True, True]
d_omode = ["per", "per", "per"]
d_cmode = "hwc"

### initialize dataset
qprint("initializing dataset", 2)
dataset = datacube_loader(dataset, folding, d_shuffle, d_batch, d_xref, d_yref, d_meanstd, d_mmode, d_omode, d_cmode)
qprint("done initializing dataset", 2)

cross_folds = dataset.k_folds
dataset.summarize()

### training parameters
models = []
model_hparams = []
save_names = []

