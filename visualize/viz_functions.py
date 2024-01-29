from matplotlib import pyplot as plt
import numpy as np
import sys
import math
sys.path.append('../models_3')
import model_bank as mb
from data_handler import data_wrangler

def dict_reformat(dicts, key):
    ddata = []
    for elt in dicts:
        print(elt.keys())
        ddata.append(elt[key])
    return ddata

### make comparison table
def viz_table(tablemetrics, tablets, savename):
    ### expected format [model][folds][data]
    ### expected format [model][fold][timeseries]
    datnames = ["mse_wue",
                "mse_esi",
                "mse_agb",
                "mae_wue",
                "mae_esi",
                "mae_agb",
                "time"]
    actualtime = []
    bigtable = [[] for i in range(len(datnames))]
    for i in range(len(tablets)):
        actualtime.append(tablets[i][-1] - tablets[i][0])

    for i in range(len(tablemetrics)):
        for j in range(6):
            bigtable[j].append(np.mean(tablemetrics[i][j]))
        bigtable[6].append(actualtime[i])
    print(bigtable)
    with open(savename, 'w+') as bigfile:
        for i in range(len(bigtable)):
            linestr = ""
            for j in range(len(bigtable[i])-1):
                linestr += str(bigtable[i][j]) + ","
            linestr += str(bigtable[i][-1]) + "\n"
            bigfile.write(linestr)

def viz_table_min(tablemetrics, tablets, savename):
    ### expected format [model][folds][data]
    ### expected format [model][fold][timeseries]
    datnames = ["mse_wue",
                "mse_esi",
                "mse_agb",
                "time"]
    actualtime = []
    bigtable = [[] for i in range(len(datnames))]
    for i in range(len(tablets)):
        actualtime.append(tablets[i][-1] - tablets[i][0])

    ### get min over training
    ### over models
    for i in range(len(tablemetrics)):
        for j in range(3):
            bigtable[j].append(np.min(tablemetrics[i][j]))
        bigtable[3].append(actualtime[i])

    print(bigtable)
    print(bigtable)
    with open(savename, 'w+') as bigfile:
        for i in range(len(bigtable)):
            linestr = ""
            for j in range(len(bigtable[i])-1):
                linestr += str(bigtable[i][j]) + ","
            linestr += str(bigtable[i][-1]) + "\n"
            bigfile.write(linestr)


### visualize training loss - over splits
def viz_training_folds(vizdata, groups, gcolors, labels, vtitle, axtitles, saveto, mode="folds"):
    ### expected format: [folds][timeseries] or [models][folds][timeseries]
    if mode == "folds":
        for i in range(len(vizdata)):
            vizdata[i] = np.array(vizdata[i])
    else:
        #need to average over folds
        for i in range(len(vizdata)):
            vizdata[i] = np.array(vizdata[i]).mean(axis=0)

    apply_colors = [gcolors[groups[ii]] for ii in range(len(groups))]
    fig, ax = plt.subplots(figsize=(12, 8))
    for pltline in range(len(vizdata)):
        ax.plot(np.arange(len(vizdata[pltline])), vizdata[pltline], label=labels[pltline],
                color=apply_colors[pltline])
    ax.set(xlabel=axtitles[0], ylabel=axtitles[1], title=vtitle)
    plt.legend(title='Model')
    plt.savefig(saveto)

def viz_training_models():
    ### expected format: (models, folds, timeseries)
    pass

def viz_training_metrics(data, groups, gcolors, labels, vtitle, saveto):
    ### expected data format: [model][fold](y/yhat shape)
    ### [model]
    ### [model]

    ### [model][dist]

    bar_mean = []
    for i in range(len(data)):
        data[i] = np.array(data[i])
        data[i] = data[i].mean(axis=tuple(range(1, data[i].ndim))).flatten()
        bar_mean.append(data[i].mean())

    print(bar_mean)
    print(data)
    alignto = np.arange(1, len(data)+1)
    apply_colors = [gcolors[groups[ii]] for ii in range(len(groups))]
    fig, ax = plt.subplots()
    ax.bar(alignto, bar_mean, color=apply_colors)#, labels=labels)
    ax.boxplot(data, vert=True, widths=(0.4), labels=labels)
    ax.set_title(vtitle)
    plt.savefig(saveto)


def viz_fold_y_yhat():
    ### per fold: y, yhat
    pass

def viz_fraction_under(data, groups, gcolors, labels, vtitle, saveto, axtitles, linspace_size=1000, avg=False):
    ### data - either [model][fold][data] or [fold][data]
    ### comparison 1 - folds within model
    ### comparison 2 - fold avgs vs other models
    construction = np.linspace(0, 1, linspace_size)
    fracs_under_data = []
    for i in range(len(data)):
        data[i] = np.array(data[i])
        frac_under_step = np.zeros(construction.shape)
        if avg:
            data[i] = data[i].mean(axis=0).flatten()
            ### ^ now averaged... compute unders
        else:
            data[i].flatten()
        for k in range(len(construction)):
            frac_under_step[k] = len(np.where(data[i] < construction[k])[0])/len(data[i])
        ### ^ fraction where eg. mse < whatever point
        ### ideally should be 1 the entire time if perfect
        fracs_under_data.append(frac_under_step)

    apply_colors = [gcolors[groups[ii]] for ii in range(len(groups))]
    fig, ax = plt.subplots()
    for pltline in range(len(data)):
        ax.plot(construction, fracs_under_data[pltline], label=labels[pltline], color=apply_colors[pltline])
    ax.set(xlabel=axtitles[0], ylabel=axtitles[1], title=vtitle)
    plt.legend(title='Parameter where:')
    plt.savefig(saveto)

def viz_overhead(saveto):
    group = ["Pyramid", "Adjusted", "Cube"]
    data_sample = [93.89731400000005,187.30869700000005,196.97674999999998]
    data_total = [484.26482000000004,592.500166,704.025192]
    #data_storage = [,]
    title = "Dataset Formatting Overhead"
    xlabel = "Method"
    ylabel = "Process Time"

    fig, ax = plt.subplots()
    ax.bar(group, data_total, width=-0.4, align='edge', color="orange", label="Total Pipeline Time")
    ax.bar(group, data_sample, width=0.4, align='edge', color="skyblue", label="Sampling Time")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.legend()
    plt.savefig(saveto)

def viz_storage(saveto):
    group = ["Pyramid", "Adjusted", "Cube"]
    data_disc = [4.01,4.02,6.24]
    data_params = [13.897,13.851,21.964]
    #data_storage = [,]
    title = "Dataset Storage Costs"
    xlabel = "Method"
    ylabel = "Size"

    fig, ax = plt.subplots()
    ax.bar(group, data_disc, width=-0.4, align='edge', color="salmon", label="Total Dataset Size (GB)")
    ax.bar(group, data_params, width=0.4, align='edge', color="lightgreen", label="Parameters Per Sample (Thousands)")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.legend()
    plt.savefig(saveto)

def viz_geo(modeldir, loadername, modifier, modeltype, dataset, batchsize, fold, focusy, var,
            saveto):
    ### SETUP ---\
    from osgeo import gdal
    ### need to redo predictions
    model = mb.make_model(modeltype=modeltype)
    #self.modeldir + "/model_" + self.name

    metacols = []
    with open("../data/" + dataset + "/info.txt", 'r') as infofile:
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

    model_parameters = {"cascade1": {"training_loss": "mse",
                                     "monitor_loss": True},
                        "flat1": {"training_loss": "mse",
                                  "monitor_loss": True},
                        "cascade2": {"training_loss": "mse",
                                     "monitor_loss": True,
                                     "variant": var,
                                     "singletask": None},
                        "flat2": {"training_loss": "mse",
                                  "monitor_loss": True},
                        }
    core_parameters = {"verbosity": 0, "dir": modeldir + "/" + loadername, "layerdims": layer_dims,
                       "x_layers": x_layers,
                       "y_layers": y_layers}

    model.setup(model_parameters[modeltype] | core_parameters, loadername + "_" + str(modifier))
    model.load()

    va_wrangler = data_wrangler("../data/" + dataset, n_layers, n_folds, layer_dims, batchsize,
                                buffer_nodata, x_layers, y_layers)
    va_wrangler.set_fold(fold)
    data_coords = np.genfromtxt("../data/" + dataset + "/legal_ids.csv", delimiter=',')
    cutoff = 100
    #val_ids = va_wrangler.val_ids[0]
    val_ids = va_wrangler.val_ids[0]#[:cutoff]

    ### END SETUP

    layer_raster_locs = ["../data/raster/ecostresswue_clipped_co.tif",
                         "../data/raster/ecostressesi_clipped_co.tif",
                         "../data/raster/gedi_agforestbiomass_clipped_co.tif"]
    layers_nd = []
    layers_s = []
    layers_gtf = []
    layers_gproj = []
    layers_data = []
    for lrl in layer_raster_locs:
        layer_raster = gdal.Open(lrl)
        rasterband = layer_raster.GetRasterBand(1)
        layers_nd.append(rasterband.GetNoDataValue())
        layers_s.append((layer_raster.RasterXSize, layer_raster.RasterYSize))
        layers_gtf.append(layer_raster.GetGeoTransform())
        layers_gproj.append(layer_raster.GetProjection())
        # tpxv = abs(tpxv)
        layers_data.append(layer_raster.ReadAsArray().transpose())
        del rasterband
        del layer_raster



    ### start with img...
    chief_res = 1000
    y_dim_options = [70, 70, 1000]
    chief_dims = layers_data[2].shape
    sample_dims = math.ceil(chief_res / y_dim_options[focusy])

    model.y_ids = y_layers
    ys, yhats = model.predict(va_wrangler)
    ys = ys[focusy]#[:cutoff]
    yhats = yhats[focusy]#[:cutoff]
    mse = (yhats - ys) ** 2
    #mse = mse.mean(axis=(1, 2))

    chief_dims = (int((chief_res / y_dim_options[focusy]) * chief_dims[0] + 10),
                  int((chief_res / y_dim_options[focusy]) * chief_dims[1] + 10))

    empty = np.zeros(chief_dims)
    empty.fill(float("nan"))

    for i in range(len(val_ids)):
        tempc = data_coords[val_ids[i].astype(int)]
        # print(tempc)
        # convert from chief units to current scale
        uli = int((chief_res / y_dim_options[focusy]) * tempc[0])
        ulj = int((chief_res / y_dim_options[focusy]) * tempc[1])
        if i == 1:
            print(uli, ulj)
        empty[int(uli):int(uli) + sample_dims, int(ulj):int(ulj) + sample_dims] = mse[i]

    #surgery on gtf
    #tulh, tpxh, _, tulv, _, tpxv =
    layers_gtf[2] = (layers_gtf[2][0], layers_gtf[2][1] * (y_dim_options[focusy] / chief_res),
                     layers_gtf[2][2], layers_gtf[2][3], layers_gtf[2][4],
                     layers_gtf[2][5] * (y_dim_options[focusy] / chief_res))
    driver = gdal.GetDriverByName("GTiff")
    print(saveto)
    layer_out = driver.Create(saveto, empty.shape[0], empty.shape[1], 1, gdal.GDT_Float32)
    layer_out.SetGeoTransform(layers_gtf[2])
    layer_out.SetProjection(layers_gproj[2])
    layer_out.GetRasterBand(1).WriteArray(empty.transpose())
    layer_out.GetRasterBand(1).SetNoDataValue(float("nan"))
    layer_out.FlushCache()
    print("saved")

def viz_heatmap(modeldir, loadername, modifier, modeltype, dataset, batchsize, fold, focusy, var,
                saveto):
    model = mb.make_model(modeltype=modeltype)
    with open("../data/" + dataset + "/info.txt", 'r') as infofile:
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

    model_parameters = {"cascade1": {"training_loss": "mse",
                                     "monitor_loss": True},
                        "flat1": {"training_loss": "mse",
                                  "monitor_loss": True},
                        "cascade2": {"training_loss": "mse",
                                     "monitor_loss": True,
                                     "variant": var,
                                     "singletask": None},
                        "flat2": {"training_loss": "mse",
                                  "monitor_loss": True},
                        }
    core_parameters = {"verbosity": 0, "dir": modeldir + "/" + loadername, "layerdims": layer_dims,
                       "x_layers": x_layers,
                       "y_layers": y_layers}

    model.setup(model_parameters[modeltype] | core_parameters, loadername + "_" + str(modifier))
    model.load()

    va_wrangler = data_wrangler("../data/" + dataset, n_layers, n_folds, layer_dims, batchsize,
                                buffer_nodata, x_layers, y_layers)
    va_wrangler.set_fold(fold)
    ys, hs = model.predict(va_wrangler)
    ys = ys[focusy].flatten()
    hs = hs[focusy].flatten()

    fig, ax = plt.subplots()

    im = ax.hist2d(ys, hs, bins=200)
    ax.figure.colorbar(im[3], ax=ax)
    ax.set(xlabel="True Value", ylabel="Predicted Value", title="True vs Predicted Value Heatmap")
    plt.savefig(saveto + ".png")

    plt.clf()

    fig, ax = plt.subplots()
    im2 = ax.hist2d(ys, hs, bins=200, norm="log")
    ax.figure.colorbar(im2[3], ax=ax)
    ax.set(xlabel="True Value", ylabel="Predicted Value", title="True vs Predicted Value Log Heatmap")
    plt.savefig(saveto + "_log.png")

    plt.clf()

    plt.scatter(ys, hs, alpha=0.05)
    plt.savefig(saveto + "_scatter.png")

def viz_atval(modeldir, loadername, modifier, modeltype, dataset, batchsize, fold, focusy, var,
                saveto):
    print("making", modeltype)
    model = mb.make_model(modeltype=modeltype)
    with open("../data/" + dataset + "/info.txt", 'r') as infofile:
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

    model_parameters = {"cascade1": {"training_loss": "mse",
                                     "monitor_loss": True},
                        "flat1": {"training_loss": "mse",
                                  "monitor_loss": True},
                        "cascade2": {"training_loss": "mse",
                                     "monitor_loss": True,
                                     "variant": var,
                                     "singletask": None},
                        "flat2": {"training_loss": "mse",
                                  "monitor_loss": True},
                        }
    core_parameters = {"verbosity": 0, "dir": modeldir + "/" + loadername, "layerdims": layer_dims,
                       "x_layers": x_layers,
                       "y_layers": y_layers}

    model.setup(model_parameters[modeltype] | core_parameters, loadername + "_" + str(modifier))
    model.load()

    va_wrangler = data_wrangler("../data/" + dataset, n_layers, n_folds, layer_dims, batchsize,
                                buffer_nodata, x_layers, y_layers)
    va_wrangler.set_fold(fold)
    ys, hs = model.predict(va_wrangler)
    ys = ys[focusy].flatten()
    hs = hs[focusy].flatten()
    ybounds = [np.min(ys), np.max(ys)]
    nbins = 200
    yside = [[] for ii in range(len(nbins) + 1)]
    yavgs = [[] for ii in range(len(nbins) + 1)]
    mse = (hs - ys) ** 2
    for i in range(len(ys)):
        y_int = int(((ys[i] - ybounds[0]) / ybounds[1]) * 200)
        yside[y_int].append(i)

    for i in range(len(yside)):
        yavgs[i] = [mse[yside[i][kk]] for kk in range(len(yside[i]))]
        yavgs[i] = np.array(yavgs[i])
        if len(yside[i]) == 0:
            yavgs[i] = 0


    """ybounds = [np.min(ys), np.max(ys)]
    yhatbounds = [np.min(hs), np.max(hs)]
    nbins = 200
    heatmap = np.zeros((nbins+1, nbins+1))
    for i in range(len(ys)):
        ### (ybounds[1] - ybounds[0]) / 200 <- step
        yhint = int(((hs[i] - yhatbounds[0])/yhatbounds[1]) * 200)
        y_int = int(((ys[i] - ybounds[0]) / ybounds[1]) * 200)
        heatmap[yhint, y_int] += 1

    yhticks = np.linspace(yhatbounds[0], yhatbounds[1], 8)
    y_ticks = np.linspace(ybounds[0], ybounds[1], 8)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.set_xticklabels(yhticks)
    plt.set_yticklabels(y_ticks)
    plt.title("2-D Heat Map")
    plt.show()"""

