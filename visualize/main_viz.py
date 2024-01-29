import viz_functions as vf
import pickle
import sys

model_dir = "../models_3/trained/"
### name, nfolds
#loader = [["test3f", 3]]
loader = [["c2a_pyramid_e1_200", 10],   #0
          ["c2b_pyramid_e1_200", 10],   #1
          ["c2c_pyramid_e1_200", 10],
          ["c2a_cube_e1_200", 10],
          ["c2a_pac_e1_200", 10],
          ["f2pac_e1_200", 10],
          ["c2a_single_wue_400", 10],   #6
          ["c2a_single_esi_400", 10],   #7
          ["c2a_single_agb_400", 10],   #8
          ["test_c2pac250", 10],        # c2d pac
          ["f2new_1_250", 10],          #10
          ["c2d_pac1", 10]]
lm = []
ll = []

loadfolds = [1]

### "/metric_" + str(fold) + ".txt"
for elt in loader:
    with open(model_dir + elt[0] + "/metric_" + elt[0] + ".txt", "rb") as metricfile:
        lm.append(pickle.load(metricfile))
    ll.append([])
    for i in loadfolds:
        with open(model_dir + elt[0] + "/cblog_" + elt[0] + "_" + str(i) + ".txt", "rb") as lossfile:
            ll[-1].append(pickle.load(lossfile))


print("keys")
print(ll[1][0].keys())

run_viz = [int(sys.argv[1])]

vcolors = ["blue", "pink", "orange", "green", "lightgray", "red", "lightblue",
           "gray", "brown", "yellow"]

### visualization 0 - bar/whiskers over folds
if 0 in run_viz:
    v0_data = [lm[0]["mse_single"][0],
               lm[1]["mse_single"][0],
               lm[2]["mse_single"][0],
               lm[3]["mse_single"][0]]
    v0_groups = [0, 1, 2, 3]
    v0_labels = ["Cascade Pyramid WUE MSE", "Flat Pyramid WUE MSE",
                 "Flat Cube WUE MSE", "Flat PACube WUE MSE"]
    v0_title = "10-fold Model WUE MSE Comparison"
    v0_saveto = "figures/f10_mse_Comparison_1_wue"
    vf.viz_training_metrics(v0_data, v0_groups, vcolors, v0_labels, v0_title, v0_saveto)
if 3 in run_viz:
    v0_data = [lm[0]["mae_single"][0],
               lm[1]["mae_single"][0],
               lm[2]["mae_single"][0],
               lm[3]["mae_single"][0]]
    v0_groups = [0, 1, 2, 3]
    v0_labels = ["Cascade Pyramid WUE MAE", "Flat Pyramid WUE MAE",
                 "Flat Cube WUE MAE", "Flat PACube WUE MAE"]
    v0_title = "10-fold Model WUE MAE Comparison"
    v0_saveto = "figures/f10_mae_Comparison_1_wue"
    vf.viz_training_metrics(v0_data, v0_groups, vcolors, v0_labels, v0_title, v0_saveto)
if 4 in run_viz:
    v0_data = [lm[0]["r2"][0],
               lm[1]["r2"][0],
               lm[2]["r2"][0],
               lm[3]["r2"][0]]
    v0_groups = [0, 1, 2, 3]
    v0_labels = ["Cascade Pyramid WUE R^2", "Flat Pyramid WUE R^2",
                 "Flat Cube WUE R^2", "Flat PACube WUE R^2"]
    v0_title = "10-fold Model WUE R^2 Comparison"
    v0_saveto = "figures/f10_mae_Comparison_1_r2"
    vf.viz_training_metrics(v0_data, v0_groups, vcolors, v0_labels, v0_title, v0_saveto)

if 5 in run_viz:
    v0_data = [lm[0]["mse_single"][1],
               lm[1]["mse_single"][1],
               lm[2]["mse_single"][1],
               lm[3]["mse_single"][1]]
    v0_groups = [0, 1, 2, 3]
    v0_labels = ["Cascade Pyramid", "Flat Pyramid",
                 "Flat Cube", "Flat PACube"]
    v0_title = "10-fold Model ESI MSE Comparison"
    v0_saveto = "figures/f10_mse_Comparison_1_esi"
    vf.viz_training_metrics(v0_data, v0_groups, vcolors, v0_labels, v0_title, v0_saveto)
if 6 in run_viz:
    v0_data = [lm[0]["mae_single"][1],
               lm[1]["mae_single"][1],
               lm[2]["mae_single"][1],
               lm[3]["mae_single"][1]]
    v0_groups = [0, 1, 2, 3]
    v0_labels = ["Cascade Pyramid", "Flat Pyramid",
                 "Flat Cube", "Flat PACube"]
    v0_title = "10-fold Model ESI MAE Comparison"
    v0_saveto = "figures/f10_mae_Comparison_1_esi"
    vf.viz_training_metrics(v0_data, v0_groups, vcolors, v0_labels, v0_title, v0_saveto)
if 7 in run_viz:
    v0_data = [lm[0]["r2"][1],
               lm[1]["r2"][1],
               lm[2]["r2"][1],
               lm[3]["r2"][1]]
    v0_groups = [0, 1, 2, 3]
    v0_labels = ["Cascade Pyramid", "Flat Pyramid",
                 "Flat Cube", "Flat PACube"]
    v0_title = "10-fold Model ESI R^2 Comparison"
    v0_saveto = "figures/f10_r2_Comparison_1_esi"
    vf.viz_training_metrics(v0_data, v0_groups, vcolors, v0_labels, v0_title, v0_saveto)

if 8 in run_viz:
    v0_data = [lm[0]["mse_single"][2],
               lm[1]["mse_single"][2],
               lm[2]["mse_single"][2],
               lm[3]["mse_single"][2]]
    v0_groups = [0, 1, 2, 3]
    v0_labels = ["Cascade Pyramid", "Flat Pyramid",
                 "Flat Cube", "Flat PACube"]
    v0_title = "10-fold Model AGB MSE Comparison"
    v0_saveto = "figures/f10_mse_Comparison_1_agb"
    vf.viz_training_metrics(v0_data, v0_groups, vcolors, v0_labels, v0_title, v0_saveto)
if 9 in run_viz:
    v0_data = [lm[0]["mae_single"][2],
               lm[1]["mae_single"][2],
               lm[2]["mae_single"][2],
               lm[3]["mae_single"][2]]
    v0_groups = [0, 1, 2, 3]
    v0_labels = ["Cascade Pyramid", "Flat Pyramid",
                 "Flat Cube", "Flat PACube"]
    v0_title = "10-fold Model AGB MAE Comparison"
    v0_saveto = "figures/f10_mae_Comparison_1_agb"
    vf.viz_training_metrics(v0_data, v0_groups, vcolors, v0_labels, v0_title, v0_saveto)
if 10 in run_viz:
    v0_data = [lm[0]["r2"][2],
               lm[1]["r2"][2],
               lm[2]["r2"][2],
               lm[3]["r2"][2]]
    v0_groups = [0, 1, 2, 3]
    v0_labels = ["Cascade Pyramid", "Flat Pyramid",
                 "Flat Cube", "Flat PACube"]
    v0_title = "10-fold Model AGB R^2 Comparison"
    v0_saveto = "figures/f10_r2_Comparison_1_agb"
    vf.viz_training_metrics(v0_data, v0_groups, vcolors, v0_labels, v0_title, v0_saveto)

### visualization 1 - percent under val
if 1 in run_viz:
    v1_data = [lm[0]["mse_single"][0],
               lm[1]["mse_single"][0],
               lm[2]["mse_single"][0],
               lm[2]["mse_single"][0]]
    v1_groups = [0, 1, 2, 3]
    v1_labels = ["Cascade Pyramid", "Flat Pyramid",
                 "Flat Cube", "Flat PACube"]
    v1_title = "Flat1 Pyramid Fraction of Points Below MSE WUE avg"
    v1_saveto = "figures/f10_comparison_mse_frac_curve.png"
    v1_axtitles = ["MSE Value", "Fraction of Points Below MSE Value"]
    vf.viz_fraction_under(v1_data, v1_groups, vcolors, v1_labels, v1_title, v1_saveto,
                          v1_axtitles, avg=True)

if 2 in run_viz:
    v2_data = [vf.dict_reformat(ll[1], "val_wue_loss"),
               vf.dict_reformat(ll[2], "val_wue_loss"),
               vf.dict_reformat(ll[3], "val_wue_loss")]
    v2_groups = [0, 1, 2]
    v2_labels = ["WUE MSE 1", "WUE MSE 2", "WUE MSE 3"]
    v2_title = "training curves"
    v2_saveto = "figures/test3f_pyramid_training_fold_time.png"
    v2_axtitles = ["Epoch", "Loss"]
    vf.viz_training_folds(v2_data, v2_groups, vcolors, v2_labels, v2_title, v2_axtitles,
                          v2_saveto, mode="comp")

if 11 in run_viz:
    v11_data = []
    v11_groups = []

    metagrps = ["c2a_pyramid", "c2b_pyramid", "c2d_pyramid", "c2a_cube", "c2a_pac", "f2_pac"]
    v11_colnames = []
    v11gc = 0
    for elt in lm:
        v11_data.append([elt["mse_single"][0],
                         elt["mse_single"][1],
                         elt["mse_single"][2],
                         elt["mae_single"][0],
                         elt["mae_single"][1],
                         elt["mae_single"][2]])
        v11gc += 1

    v11_ts = []
    for i in range(len(ll)):
        v11_ts.append(ll[i][0]["time"])

    vf.viz_table(v11_data, v11_ts, "figures/multitest_table.txt")

if 12 in run_viz:
    v2_data = []
    runon = [0, 1, 2, 3, 4, 10, 11]
    padit = []
    for i in runon:
        if i in padit:
            v2_data.append(ll[i][0]["val_esi_loss"])
        else:
            v2_data.append(ll[i][0]["val_esi_loss"][:220])

    cutoff = 210
    v2_groups = [0, 1, 2, 3, 4, 5, 6]
    v2_labels = ["C2a_pyramid", "C2b_pyramid", "C2d_pyramid", "C2a_cube", "C2a_pac",
                 "F2_pac", "C2d_pac", "C2d_cube"]
    v2_title = "Validation ESI Performance"
    v2_saveto = "figures/testmult_pyramid_training_esimse_5.png"
    v2_axtitles = ["Epoch", "ESI MSE"]
    vf.viz_training_folds(v2_data, v2_groups, vcolors, v2_labels, v2_title, v2_axtitles,
                          v2_saveto, mode="folds")

if 13 in run_viz:
    v13_onmodel = 0
    v13_variant = "a"
    v13_modeltype = "cascade2"
    v13_modifier = 1
    v13_dataset = "pyramid_sets/box_pyramid"
    v13_focus = 2
    v13_options = ["wue", "esi", "agb"]
    v13_saveto = "figures/geo_perf_" + v13_options[v13_focus] + ".tif"
    vf.viz_geo(model_dir, loader[v13_onmodel][0], v13_modifier, v13_modeltype,
               v13_dataset, 800, v13_modifier, v13_focus, v13_variant, v13_saveto)

if 14 in run_viz:
    v14_saveto = "figures/method_overhead.png"
    vf.viz_overhead(v14_saveto)

if 15 in run_viz:
    v15_saveto = "figures/method_storage.png"
    vf.viz_storage(v15_saveto)

if 16 in run_viz:
    v11_data = []
    v11_groups = []

    metagrps = ["c2a_pyramid", "c2b_pyramid", "c2d_pyramid", "c2a_cube", "c2a_pac", "f2_pac"]
    v11_colnames = []
    v11gc = 0
    runon = [11]
    for ii in runon:
        elt = ll[ii]
        v11_data.append([elt[0]["val_wue_loss"],
                         elt[0]["val_esi_loss"],
                         elt[0]["val_agb_loss"]])
        v11gc += 1

    v11_ts = []
    for i in range(len(ll)):
        v11_ts.append(ll[i][0]["time"])

    vf.viz_table_min(v11_data, v11_ts, "figures/multitest_table_min_1a.txt")

if 17 in run_viz:
    v11_data = []
    v11_groups = []

    metagrps = ["c2a_wue", "c2a_esi", "c2a_agb", "c2a_cube", "c2a_pac", "f2_pac"]
    v11_colnames = []
    v11gc = 0
    runon = [6, 7, 8]
    print(ll[runon[1]][0].keys())
    print(ll[runon[0]][0]["val_loss"][-1])
    v11_data = [[ll[runon[0]][0]["val_loss"],
                ll[runon[1]][0]["val_loss"],
                ll[runon[2]][0]["val_loss"]]]
    v11_ts = [[]]

    for i in range(len(ll[0][0]["time"])):
        v11_ts[0].append(ll[0][0]["time"][i] + ll[1][0]["time"][i] + ll[2][0]["time"][i])

    vf.viz_table_min(v11_data, v11_ts, "figures/singletest_table_min.txt")

if 18 in run_viz:
    v13_onmodel = 3
    v13_variant = "a"
    v13_modeltype = "cascade2"
    v13_modifier = 1
    v13_dataset = "pyramid_sets/box_cube"
    v13_focus = 0
    v13_options = ["wue", "esi", "agb"]
    v13_saveto = "figures/yyh_heatmap2cube2_" + v13_options[v13_focus]
    vf.viz_heatmap(model_dir, loader[v13_onmodel][0], v13_modifier, v13_modeltype,
               v13_dataset, 800, v13_modifier, v13_focus, v13_variant, v13_saveto)