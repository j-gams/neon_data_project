import numpy as np
import sys
import matplotlib.pyplot as plt

from dat_obj import datacube_loader
from datacube_set import satimg_set

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
fulltrain = np.zeros((dataset.train[0].get_n_samples() + dataset.validation[0].get_n_samples() + dataset.test.get_n_samples(), len(dataset.dchannels) + 2))

dataset.train[0].unapply_m_s()
dataset.validation[0].unapply_m_s()

ydhelper = 0
ydata_all = np.zeros(len(fulltrain))
ydata_all[:dataset.train[0].get_n_samples()] = dataset.train[0].y
ydata_all[dataset.train[0].get_n_samples():dataset.train[0].get_n_samples()+dataset.validation[0].get_n_samples()] = dataset.validation[0].y
ydata_all[dataset.train[0].get_n_samples()+dataset.validation[0].get_n_samples():] = dataset.test.y
#ydata = np.array([dataset.test.y, dataset.train[0].y, dataset.validation[0].y]).flatten()
plt.hist(ydata_all)
plt.show()