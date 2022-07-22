### Framework for managing training models

### imports...
import numpy as np
import sys

import model_train

import train_1
from custom_models import regressor_test

sys.path.insert(0, '../create_data')

from dat_obj import datacube_loader
from datacube_set import satimg_set

verbosity = 2
def qprint(pstr, importance):
    if importance <= verbosity:
        print(pstr)

### TODO - read in file params from config file... eventually

### dataset parameters
#dataset = "minidata_nosa"
dataset = "minidata"
#dataset = "data_interpolated"
#folding = "test_fold"
folding = "test_kfold"

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
#train_params = {"folds:" dataset.k_folds}
models = []
model_hparams = []
save_names = []
train_params = [{"folds": dataset.k_folds,
                 "metrics": ["mean_squared_error", "mean_absolute_error", "train_realtime", "train_processtime"],
                 "mode": "train",
                 "load_from": "na",
                 "save_models": True}]


load_list = ["train_1"]#"test_regress"]
for mdl_str in load_list:
    if mdl_str == "train_1":
        models.append(train_1.test_conv)
        model_hparams.append({"model_name": "basic_convmodel_1",
                              "save_location": "placeholder",
                              "input_size": dataset.test.dims,
                              "save_checkpoints": True,
                              "train_metric": "mean_squared_error",
                              "epochs": 50,
                              "use_best": True,
                              "save_last_epoch": True,
                              "dropout": {"mode": "drop", "channels": [2, 3]},
                              "verbosity": 1})
        save_names.append("basic_convmodel_test_1")
    elif mdl_str == "test_regress":
        models.append(regressor_test.test_regress)
        model_hparams.append({"model_name": "basic_regressor_1",
                              "save_location": "placeholder",
                              "save_checkpoints": True,
                              "train_metric": "mean_squared_error",
                              "epochs": 50,
                              "use_best": True,
                              "save_last_epoch": True,
                              "penalty": "l2",
                              "alpha": 0.0001,
                              "dropout": {"mode": "drop", "channels": [2, 3]},
                              "avg_channel": False,
                              "retain_avg": False,
                              "batch_regress": True,
                              "normalize": True,
                              "verbosity": 1})
        save_names.append("basic_sgd_regression_test_1")
    elif mdl_str == "lasso_r":
        models.append(lasso_regress.lasso_regress)
        model_hparams.append({"model_name": "lasso",
                              "save_location": "placeholder",
                              "alpha": 0.1,
                              "dropout": {"mode": "drop", "channels": [2, 3]},
                              "avg_channel": True,
                              "normalize": True,
                              "verbosity": 1})
        save_names.append("lasso_regression_1")
### now dispatch to the model trainer...?

for i in range(len(models)):
    print("sending " + save_names[i] + " to be trained")
    model_train.train(dataset, models[i], model_hparams[i], save_names[i], train_params[i])

print("done")
