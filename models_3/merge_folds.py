import pickle
import model_bank as mb

#combine = ["../models_3/trained/cascade_pyramid/metric_cascade_pyramid_a.txt",
#           "../models_3/trained/cascade_pyramid/metric_cascade_pyramid_b.txt"]
#combine = ["../models_3/trained/flat_pyramid/metric_flat1_pyramid_a.txt",
#           "../models_3/trained/flat_pyramid/metric_flat1_pyramid_b.txt"]
#combine = ["../models_3/trained/flat_cube/metric_flat1_cube_a.txt",
#           "../models_3/trained/flat_cube/metric_flat1_cube_b.txt"]
combine = ["../models_3/trained/flat_pac/metric_flat1_pac_a.txt",
           "../models_3/trained/flat_pac/metric_flat1_pac_b.txt"]
model_dir = "../models_3/trained/"
#model_name = "cascade_pyramid"
#model_name = "flat_pyramid"
#model_name = "flat_cube"
model_name = "flat_pac"
mode = "folds" # or y
cfl = []
y_layers = 3
for mdat in combine:
    with open(mdat, "rb") as metricfile:
        cfl.append(pickle.load(metricfile))

"""computed_metrics = {"metafolds": run_on_folds,
                    "mse_single": [[] for ii in range(len(y_layers))],
                    "mse_each": [[] for ii in range(len(y_layers))],
                    "mse_overall": [[] for ii in range(len(y_layers))],
                    "mae_single": [[] for ii in range(len(y_layers))],
                    "mae_each": [[] for ii in range(len(y_layers))],
                    "mae_overall": [[] for ii in range(len(y_layers))],
                    "r2": [[] for ii in range(len(y_layers))]}"""

combined = dict()
if mode == "folds":
    cfl_keys = cfl[0].keys()
    combined = {"metafolds": []}
    for k in cfl_keys:
        if k != "metafolds":
            combined[k] = [[] for ii in range(y_layers)]
    for td in cfl:
        #combined["metafolds"].extend([td["metafolds"]])
        for k in cfl_keys:
            if k == "metafolds":
                combined[k].extend(td[k])
            else:
                for i in range(y_layers):
                    combined[k][i].extend(td[k][i])

#print(cfl[0]["mse_single"][0])
print("***")
#print(combined["mse_single"][0])

mb.save_metrics(combined, model_dir + "/" + model_name, model_name)
print("done")
