### IMPORTS

### setup
### send to train...
### compile and save data
import model_bank as mb
import metrics as ms
import os
from data_handler import data_wrangler

import sys
if len(sys.argv) < 2:
    print("reminder: provide a model name")
    print("exiting...")
    sys.exit(0)
else:
    print("modelname:", sys.argv[1])

### Frame Parameters
### should have this saved with the dataset
data_rootdir = "../data/pyramid_sets/mf_test"
model_dir = "trained"
batch_size = 32
mode = "train" # {train, load, loadtrain}
override_existing_dir = True

modelname = sys.argv[1]
if not override_existing_dir and os.path.exists(model_dir + "/" + modelname):
    print("model already exists!")
    print("exiting...")
    sys.exit(0)
elif override_existing_dir and os.path.exists(model_dir + "/" + modelname):
    os.system("rm " + model_dir + "/" + modelname + "/*")

n_epochs_default = 20
if len(sys.argv) > 2:
    n_epochs = int(sys.argv[2])
else:
    n_epochs = n_epochs_default

### visualization parameters
make_vis = ["performance map"]

### Model Parameters
### model types: cascade1, flat1,
modeltype = "flat1"
verbosity = 0
model_parameters = {"cascade1": {"training_loss": "mse",
                                 "monitor_loss": True},
                    "flat1": {"training_loss": "mse",
                                 "monitor_loss": True}
                    }

### iterate over row; iterate over comma separation
### rasterloc, base_res, end_res, x/y, idx, layername
metacols = []
with open(data_rootdir + "/info.txt", 'r') as infofile:
    metatotal = infofile.read().replace('\n', ';')
metalines = metatotal.split(";")
metadata = []
for ml in metalines:
    if len(ml) > 0:
        metadata.append(ml.split(","))

other_info = metadata.pop(0)
n_folds = int(other_info[0])
buffer_nodata = int(other_info[1])
print("base data coordinate system:", other_info[2])
n_layers = len(metadata)

layer_dims = []
x_layers = []
y_layers = []
layer_names = []
for i in range(len(metadata)):
    layer_dims.append(int(metadata[i][0]))
    layer_names.append(metadata[i][3])
    if metadata[i][1] == "x":
        x_layers.append(i)
    else:
        y_layers.append(i)

for elt in metadata:
    print(elt)

core_parameters = {"verbosity": verbosity, "dir": model_dir + "/" + modelname, "layerdims": layer_dims, "x_layers": x_layers,
                   "y_layers": y_layers}

### make directory for model
os.system("mkdir " + model_dir + "/" + modelname)

### initialize wranglers
### training wrangler
tr_wrangler = data_wrangler(data_rootdir, n_layers, n_folds, layer_dims, batch_size, buffer_nodata, x_layers, y_layers)
### val wrangler
va_wrangler = data_wrangler(data_rootdir, n_layers, n_folds, layer_dims, batch_size, buffer_nodata, x_layers, y_layers)

computed_metrics = {"mse_single": [[] for ii in range(len(y_layers))],
                    "mse_each": [[] for ii in range(len(y_layers))],
                    "mse_overall": [[] for ii in range(len(y_layers))],
                    "mae_single": [[] for ii in range(len(y_layers))],
                    "mae_each": [[] for ii in range(len(y_layers))],
                    "mae_overall": [[] for ii in range(len(y_layers))],
                    "r2": [[] for ii in range(len(y_layers))]}

### for each fold...
for fold_i in range(n_folds):
    print("running fold", fold_i)
    ### build model
    model = mb.make_model(modeltype=modeltype)
    model.setup(model_parameters[modeltype] | core_parameters, modelname + "_" + str(fold_i))
    ### do load/start from scratch...
    tr_wrangler.set_fold(fold_i)
    va_wrangler.set_fold(fold_i)
    tr_wrangler.set_mode("train")
    va_wrangler.set_mode("val")
    if mode == "load" or mode == "loadtrain":
        model.load()
    if mode == "train" or mode == "loadtrain":
        print("fitting", fold_i)
        model.fit(tr_wrangler, va_wrangler, n_epochs=n_epochs)
        model.save()
        ys, hs = model.predict(va_wrangler)

        ### compute metrics


        for yl in range(len(y_layers)):
            #mse_s, mse_e, mse_o = ms.metric_mse(ys[yl], hs[yl], "geo", ["single", "each", "overall"])
            #mae_s, mae_e, mae_o = ms.metric_mae(ys[yl], hs[yl], "geo", ["single", "each", "overall"])

            mse_s = ms.metric_mse(ys[yl], hs[yl], "geo", ["single"])
            mae_s = ms.metric_mae(ys[yl], hs[yl], "geo", ["single"])

            computed_metrics["mse_single"][yl].append(mse_s)
            #computed_metrics["mse_each"][yl].append(mse_e)
            #computed_metrics["mse_overall"][yl].append(mse_o)
            computed_metrics["mae_single"][yl].append(mae_s)
            #computed_metrics["mae_each"][yl].append(mae_e)
            #computed_metrics["mae_overall"][yl].append(mae_o)

            computed_metrics["r2"][yl].append(ms.metric_r2(ys[yl], hs[yl]))

mb.save_metrics(computed_metrics, model_dir + "/" + modelname, modelname)

