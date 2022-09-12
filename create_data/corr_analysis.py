import numpy as np
import sys
sys.path.insert(0, '../create_data')
from dat_obj import datacube_loader
from datacube_set import satimg_set
import matplotlib.pyplot as plt

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

dataset = datacube_loader(dataset, folding, d_shuffle, d_batch, d_xref, d_yref, d_meanstd, d_mmode, d_omode, d_cmode)

cross_folds = dataset.k_folds
dataset.summarize()

### entire set is train union test union val

fulltrain = np.zeros((dataset.train[0].get_n_samples() + dataset.validation[0].get_n_samples() + dataset.test.get_n_samples(),
                      len(dataset.dchannels) + 2))

dataset.train[0].set_flatten(True)
dataset.validation[0].set_flatten(True)
dataset.test.set_flatten(True)

fully = []
sidx = 0
for i in range(len(dataset.train[0])):
    xvals, yvals = dataset.train[0][i]
    #print(xvals.shape)
    for j in range(len(yvals)):
        for k in range(68):
            ttemp = np.mean(xvals[j, [(ii*68) + k for ii in range(xvals.shape[1]//68)]])
            #print(ttemp)
            fulltrain[sidx,k] = ttemp #np.mean(xvals[j, [(ii*68) + i for ii in range(xvals.shape[1]//68)]])
        #fulltrain[sidx] = np.mean(xvals[j], axis=1)
        fully.append(yvals[j])
        sidx += 1

for i in range(len(dataset.validation[0])):
    xvals, yvals = dataset.validation[0][i]
    for j in range(len(yvals)):
        for k in range(68):
            fulltrain[sidx,k] = np.mean(xvals[j, [(ii*68) + k for ii in range(xvals.shape[1]//68)]])
        #fulltrain[sidx] = np.mean(np.xvals[j], axis=1)
        fully.append(yvals[j])
        sidx += 1

for i in range(len(dataset.test)):
    xvals, yvals = dataset.test[i]
    for j in range(len(yvals)):
        for k in range(68):
            fulltrain[sidx,k] = np.mean(xvals[j, [(ii*68) + k for ii in range(xvals.shape[1]//68)]])
        #fulltrain[sidx] = np.mean(np.xvals[j], axis=1)
        fully.append(yvals[j])
        sidx += 1

npy = np.array(fully)
print(fulltrain)
dlimit = -100000
ulimit = 100000
nfound = 0
for i in range(fulltrain.shape[0]):
    for j in range(fulltrain.shape[1]):
        if fulltrain[i, j] > ulimit or fulltrain[i, j] < dlimit:
            fulltrain[i, j] = float("nan")
            nfound += 1
print("do the dew:", nfound, "found")
for featn in range(len(dataset.dchannels)):
    plt.figure()
    plt.scatter(fulltrain[:,featn].flatten(), npy)
    plt.title(dataset.dchannels[featn][0] + " vs ecostress WUE values")
    plt.xlabel(dataset.dchannels[featn][0])
    plt.ylabel("ecostress WUE")
    plt.savefig("../figures/corr_comparison/" + dataset.dchannels[featn][0] + "_" + str(featn) + ".png")
    plt.cla()
    plt.close()
