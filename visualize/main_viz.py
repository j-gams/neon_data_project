import viz_functions as vf
import pickle

model_dir = "../models_3/trained/"
### name, nfolds
loader = [["test3f", 3]]
lm = []
ll = []

### "/metric_" + str(fold) + ".txt"
for elt in loader:
    with open(model_dir + elt[0] + "/metric_" + elt[0] + ".txt", "rb") as metricfile:
        lm.append(pickle.load(metricfile))
    ll.append([])
    for i in range(elt[1]):
        with open(model_dir + elt[0] + "/cblog_" + elt[0] + "_" + str(i) + ".txt", "rb") as lossfile:
            ll[-1].append(pickle.load(lossfile))

run_viz = [1]

vcolors = ["blue", "pink", "orange", "lightgray", "yellow"]

### visualization 1 - bar/whiskers over folds
if 1 in run_viz:
    v1_data = [lm[0]["mse_single"],
               lm[0]["mae_single"],
               lm[0]["r2"]]
    v1_groups = [0, 0, 1]
    v1_labels = ["MSE", "MAE", "R^2"]
    v1_title = "Flat1 Pyramid Metrics Side-By-Side"
    v1_saveto = "figures/test3f_pyramid_metrics_comparison.png"
    vf.viz_training_metrics(v1_data, v1_groups, vcolors, v1_labels, v1_title, v1_saveto)