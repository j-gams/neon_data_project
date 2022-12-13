### Written by Jerry Gammie @j-gams

### Framework for managing training models

### imports...
import numpy as np
import sys

import model_train

import train_1
import train_3
import train_noise
from custom_models import regressor_test
from custom_models import auto_regress
from custom_models import lasso_regress
from custom_models import kernel_regress
from custom_models import reduce_regress
from custom_models import svr_1
from custom_models import rf_regress
from custom_models import gradientboosting_regress
from custom_models import cnn_basic
from custom_models import cnn_blocks
from custom_models import t1
sys.path.insert(0, '../create_data')

from dat_obj import datacube_loader
from datacube_set import satimg_set

verbosity = 2
def qprint(pstr, importance):
    if importance <= verbosity:
        print(pstr)

### TODO - read in file params from config file... eventually
run_h5mode = False
if sys.argv[1] == "1":
    dataset = "minidata"
    folding = "test_kfold"
elif sys.argv[1] == "2":
    dataset = "data_h51"
    folding = "fold_2"
else:
    dataset = "data_interpolated"
    folding = "test_fold"
override_mdl = None
if len(sys.argv) >= 3:
    override_mdl = sys.argv[2]
if len(sys.argv) == 4:
    if sys.argv[3] == "h5":
        print("running in h5 mode")
        run_h5mode = True
### dataset parameters
#dataset = "minidata_nosa"
#dataset = "minidata"
#dataset = "data_interpolated"
#folding = "test_fold"
#folding = "test_kfold"

d_batch = 12
d_shuffle = [True, True, True]
d_batch = [d_batch, d_batch, d_batch]
d_meanstd = ["default", "default", "default"]
### TODO - move these to meta file
d_xref = 1
d_yref = 2
d_h5ref = 0
d_mmode = [True, True, True]
d_omode = ["per", "per", "per"]
d_cmode = "hwc"
d_h5mode = run_h5mode

### initialize dataset
qprint("initializing dataset", 2)
dataset = datacube_loader(dataset, folding, d_shuffle, d_batch, d_xref, d_yref, d_h5ref, d_meanstd, d_mmode, d_omode, d_cmode, d_h5mode)
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

if override_mdl != None:
    load_list = [override_mdl]
else:
    load_list = ["cnn1"]
for mdl_str in load_list:
    if mdl_str == "cnn":
        models.append(cnn_basic.cnn)
        model_hparams.append({"model_name": "basic_cnn",
                              "save_location": "placeholder",
                              "input_size": dataset.test.dims,
                              "save_checkpoints": True,
                              "train_metric": "mean_squared_error",
                              "epochs": 100,
                              "use_best": True,
                              "save_last_epoch": True,
                              "dropout": {"mode": "drop", "channels": [66, 67]},
                              #"noise": 0.001,
                              "arch": [["conv2d", [128, (3,3), 2, 'same', 'relu'],  1],
                                       ["batchnorm", None, 1],
                                       ["maxpooling2d", [2, 2], 1],
                                       ["conv2d", [256, (3,3), 2, 'same', 'relu'], 1],
                                       ["batchnorm", None, 1],
                                       ["maxpooling2d", [2,2], 1],
                                       ["conv2d", [512, (3,3), 1, 'same', 'relu'], 1],
                                       ["batchnorm", None, 1],
                                       ["conv2d", [1028, (3,3), 1, 'same', 'relu'], 1],
                                       ["batchnorm", None, 1],
                                       ["flatten", None, 1],
                                       ["dense", [1028, 'relu', None], 1],
                                       ["batchnorm", None, 1],
                                       ["dense", [512, 'relu', None], 1],
                                       ["batchnorm", None, 1],
                                       ["dense", [200, 'relu', None], 1],
                                       ["dense", [20, 'relu', None], 1]]})
        save_names.append("cnn_batchnorm_noise")
    elif mdl_str == "blocnn":
        models.append(cnn_blocks.cnn_block)
        model_hparams.append({"model_name": "bloc_cnn",
                              "save_location": "placeholder",
                              "input_size": dataset.test.dims,
                              "save_checkpoints": True,
                              "train_metric": "mean_squared_error",
                              "epochs": 150,
                              "use_best": True,
                              "save_last_epoch": True,
                              "dropout": {"mode": "drop", "channels": [66, 67]},
                              "arch": [["batchnorm", None],
                                       ["conv2d", [120, (3, 3), 2, 'same', 'relu']],
                                       #["maxpooling2d", [(3, 3), (2, 2)]],
                                       ["fireresidual", ["fire", True]],
                                       ["fireresidual", ["fire", True]],
                                       ["fireresidual", ["fire", True]],
                                       ["maxpooling2d", [(3, 3), (2, 2)]],
                                       ["fireresidual", ["fire", True]],
                                       ["fireresidual", ["fire", True]],
                                       ["fireresidual", ["fire", True]],
                                       ["fireresidual", ["fire", True]],
                                       #["maxpooling2d", [(3, 3), (2, 2)]],
                                       ["fireresidual", ["fire", True]],
                                       ["conv2d", [720, (3, 3), 2, "same", "relu"]],
                                       ["globalavg2d", None],
                                       ["flatten", None],
                                       ["dense", [720, 'relu', None]],
                                       ["dense", [400, 'relu', None]],
                                       ["dense", [1, 'relu', None]]],
                              "metaparams": {"fire": [200, 0.45, 0.4, 2, 120]}})
        save_names.append("cnn_fire")
    elif mdl_str == "mininn":
        models.append(cnn_blocks.cnn_block)
        model_hparams.append({"model_name": "mini_nn",
                              "save_location": "placeholder",
                              "input_size": dataset.test.dims,
                              "save_checkpoints": True,
                              "train_metric": "mean_squared_error",
                              "epochs": 150,
                              "use_best": True,
                              "save_last_epoch": True,
                              "dropout": {"mode": "none", "channels": [66, 67]},
                              "arch": [["flatten", None],
                                       ["batchnorm", None],
                                       ["dense", [720, 'relu', None]],
                                       ["batchnorm", None],
                                       ["dense", [800, 'relu', None]],
                                       ["batchnorm", None],
                                       ["dense", [800, 'relu', None]],
                                       ["batchnorm", None],
                                       ["dense", [720, 'relu', None]],
                                       ["batchnorm", None],
                                       ["dense", [500, 'relu', None]],
                                       ["batchnorm", None],
                                       ["dense", [500, 'relu', None]],
                                       ["batchnorm", None],
                                       ["dense", [20, 'relu', None]],
                                       ["batchnorm", None],
                                       ["dense", [1, 'relu', None]]]})
        save_names.append("cnn_fire")
    elif mdl_str == "t1":
        models.append(t1.t1test)
        model_hparams.append({"model_name": "transformer_cnn",
                              "save_location": "placeholder",
                              "input_size": dataset.test.dims,
                              "save_checkpoints": True,
                              "train_metric": "mean_squared_error",
                              "epochs": 100,
                              "use_best": True,
                              "save_last_epoch": True,
                              "dropout": {"mode": "drop", "channels": [66, 67]},
                              "noise": 0.001})
        save_names.append("transformer1")
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
                              "dropout": {"mode": "keep", "channels": [0, 1, 2, 3]},
                              #"dropout": {"mode": "keep", "channels": [0, 1]},
                              "avg_channel": True,
                              "retain_avg": True,
                              "batch_regress": False,
                              "normalize": False,
                              "verbosity": 1})
        save_names.append("basic_linreg_all")
    elif mdl_str == "lasso_r":
        models.append(lasso_regress.lasso_regress)
        model_hparams.append({"model_name": "lasso",
                              "save_location": "placeholder",
                              "alpha": 0.2,
                              "dropout": {"mode": "keep", "channels": [0, 1, 2, 3]},
                              "avg_channel": True,
                              "normalize": True,
                              "verbosity": 1})
        save_names.append("lasso_h5_a02_allchannels")
    elif mdl_str == "kernel_r":
        models.append(kernel_regress.kernel_regress)
        model_hparams.append({"model_name": "lasso",
                              "save_location": "placeholder",
                              "alpha": 0.2,
                              "kernel": "ply",
                              "dropout": {"mode": "keep", "channels": [0, 1, 2, 3]},
                              "avg_channel": True,
                              "normalize": True,
                              "verbosity": 1})
        save_names.append("kernel_regression_rbf_1")
    elif mdl_str == "auto_r":
        models.append(auto_regress.test_auto)
        model_hparams.append({"model_name": "basic_autor_1",
                              "save_location": "placeholder",
                              "input_size": dataset.test.dims,
                              "save_checkpoints": True,
                              "train_metric": "binary_crossentropy",
                              "epochs": 20,
                              "use_best": True,
                              "save_last_epoch": True,
                              "dropout": {"mode": "drop", "channels": [2, 3]},
                              "encoding_size": 8,
                              "denselayers": [1024, 512, 256],
                              "rstep": "kerr",
                              "rstep_params": {"alpha": 0.2},
                              "verbosity": 1})
        save_names.append("autoencoder_rbfk_reg_1_5dim")
    elif mdl_str == "reduce_r":
        models.append(reduce_regress.decompose_regress)
        model_hparams.append({"model_name": "decompose_regress",
                              "save_location": "placeholder",
                              "dropout": {"mode": "none", "channels": [0, 1, 2, 3]},
                              "reduce_to": 5,
                              "rmode": "lr",
                              "dmode": "pca"})
        save_names.append("pca_reduce")
    elif mdl_str == "svr":
        models.append(svr_1.svregress)
        model_hparams.append({"model_name": "svr_test",
                              "save_location": "placeholder",
                              "dropout": {"mode": "keep", "channels": [0, 1, 2, 3]},
                              "kernel": "poly",
                              "poly_degree": 5,
                              "gamma": "scale"})
        save_names.append("svr_t1")
    elif mdl_str == "rfr":
        models.append(rf_regress.rfregressor)
        model_hparams.append({"model_name": "rfr_test",
                              "save_location": "placeholder",
                              "dropout": {"mode": "keep", "channels": [0, 1, 2, 3]},
                              "n_estimators": 1000,
                              "max_depth": None,
                              "n_jobs": -1})
        save_names.append("rfr_t1")
    elif mdl_str == "gbr":
        models.append(gradientboosting_regress.gbregressor)
        model_hparams.append({"model_name": "gbr_test",
                              "save_location": "placeholder",
                              "dropout": {"mode": "drop", "channels": [0, 1, 2, 3, 66, 67]},
                              "n_estimators": 1000,
                              "max_depth": 3,
                              "learning_rate": 0.1})
        save_names.append("gradientboosting_t2_both")
### now dispatch to the model trainer...

for i in range(len(models)):
    print("sending " + save_names[i] + " to be trained")
    model_train.train(dataset, models[i], model_hparams[i], save_names[i], train_params[i])

print("done")
