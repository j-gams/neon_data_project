### Written by Jerry Gammie @j-gams

import numpy as np
import sys
from dat_obj import datacube_loader
from datacube_set import satimg_set
import matplotlib.pyplot as plt

include_test = False

dataset = "data_h51"
folding = "fold_1"

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
d_h5mode = True

dataset = datacube_loader(dataset, folding, d_shuffle, d_batch, d_xref, d_yref, d_h5ref, d_meanstd, d_mmode, d_omode, d_cmode, d_h5mode)

cross_folds = dataset.k_folds
dataset.summarize()
if not include_test:
    fulltrain = np.zeros((dataset.train[0].get_n_samples() + dataset.validation[0].get_n_samples(), len(dataset.dchannels) + 2))

dataset.train[0].unapply_m_s()
dataset.validation[0].unapply_m_s()
###
"""dataset.train[0].set_flatten(True)
dataset.validation[0].set_flatten(True)
dataset.test.set_flatten(True)"""

d_names = ["srtm", "nlcd", "slope", "aspect"]
hist_bounds = [[float("inf"), float("-inf")],
               [float("inf"), float("-inf")],
               [float("inf"), float("-inf")],
               [float("inf"), float("-inf")]]

hist_bins = np.zeros((101, 4))
print("running")
encounters = 0
enc_count = [0, 0, 0, 0]
mins = [10000, 10000, 10000, 10000]
maxs = [0, 0, 0, 0]
for j in range(len(dataset.train[0])):
    xset = dataset.train[0][j][0]
    for i in range(4):
        for k in range(xset.shape[0]):
            for ii in range(16):
                for ij in range(16):
                    if xset[k, ii, ij, i] > hist_bounds[i][1]:
                        hist_bounds[i][1] = xset[k, ii, ij, i]
                    if xset[k, ii, ij, i] < hist_bounds[i][0]:
                        hist_bounds[i][0] = xset[k, ii, ij, i]


                    #if xset[k, ii, ij, i] > maxs[i]:
                    #    maxs[i] = xset[k, ii, ij, i]
                    #if xset[k, ii, ij, i] < mins[i]:
                    #    mins[i] = xset[k, ii, ij, i]
                    #hist_i = int(xset[k, ii, ij, i] / ((hist_bounds[i][1] - hist_bounds[i][0]) / 100))
                    #if hist_i >= 0 and hist_i <= 101:
                    #    hist_bins[hist_i, i] += 1
                    #else:
                    #    encounters += 1
                    #    enc_count[i] += 1

print("encounters:", encounters, enc_count)

for j in range(len(dataset.validation[0])):
    xset = dataset.validation[0][j][0]
    for i in range(4):
        for k in range(xset.shape[0]):
            for ii in range(16):
                for ij in range(16):
                    if xset[k, ii, ij, i] > hist_bounds[i][1]:
                        hist_bounds[i][1] = xset[k, ii, ij, i]
                    if xset[k, ii, ij, i] < hist_bounds[i][0]:
                        hist_bounds[i][0] = xset[k, ii, ij, i]
                    #if xset[k, ii, ij, i] > maxs[i]:
                    #    maxs[i] = xset[k, ii, ij, i]
                    #if xset[k, ii, ij, i] < mins[i]:
                    #    mins[i] = xset[k, ii, ij, i]
                    #hist_i = int(xset[k, ii, ij, i] / ((hist_bounds[i][1] - hist_bounds[i][0]) / 100))
                    #if hist_i >= 0 and hist_i <= 101:
                    #    hist_bins[hist_i, i] += 1
                    #else:
                    #    encounters += 1
                    #    enc_count[i] += 1

for j in range(len(dataset.train[0])):
    xset = dataset.train[0][j][0]
    for i in range(4):
        for k in range(xset.shape[0]):
            for ii in range(16):
                for ij in range(16):
                    hist_i = int(((xset[k, ii, ij, i] - hist_bounds[i][0])/(hist_bounds[i][1] - hist_bounds[i][0])) * 100)
                    hist_bins[hist_i, i] += 1
                    #hist_i = int(xset[k, ii, ij, i] / ((hist_bounds[i][1] - hist_bounds[i][0]) / 100))

for j in range(len(dataset.validation[0])):
    xset = dataset.validation[0][j][0]
    for i in range(4):
        for k in range(xset.shape[0]):
            for ii in range(16):
                for ij in range(16):
                    hist_i = int(((xset[k, ii, ij, i] - hist_bounds[i][0]) / (hist_bounds[i][1] - hist_bounds[i][0])) * 100)
                    hist_bins[hist_i, i] += 1

print("saving images...")
for i in range(4):
    plt.figure()
    plt.bar(np.linspace(hist_bounds[i][0], hist_bounds[i][1], 101), hist_bins[:,i])
    plt.title("Frequency of " + d_names[i] + " Over Datacube Pixels")
    plt.savefig("../figures/pixel_distributions/" + d_names[i] + "_dist.png")
    plt.xlabel(d_names[i] + " value")
    plt.ylabel("frequency")
    plt.cla()
    plt.close()

    plt.figure()
    plt.bar(np.linspace(hist_bounds[i][0], hist_bounds[i][1], 101), np.log(hist_bins[:, i]))
    plt.title("Log Frequency of " + d_names[i] + " Over Datacube Pixels")
    plt.savefig("../figures/pixel_distributions/" + d_names[i] + "_dist_log.png")
    plt.xlabel(d_names[i] + " value")
    plt.ylabel("frequency")
    plt.cla()
    plt.close()

print("report:")
print(hist_bounds)
print("done")