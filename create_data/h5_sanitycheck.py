import numpy as np
import sys

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

h5_dset = datacube_loader(dataset, folding, d_shuffle, d_batch, d_xref, d_yref, d_h5ref, d_meanstd, d_mmode, d_omode, d_cmode, True)
cv_dset = datacube_loader(dataset, folding, d_shuffle, d_batch, d_xref, d_yref, d_h5ref, d_meanstd, d_mmode, d_omode, d_cmode, False)

def within_eps(a, b, eps = 0.001):
    if abs(a-b) < eps:
        return True
    return False

print("running tests...")
h5_dset.validation[0].unshuffle()
cv_dset.validation[0].unshuffle()
tally = 0
for i in range(len(h5_dset.train[0])):
    nph5 = h5_dset.validation[0][i][0]
    npcv = cv_dset.validation[0][i][0]
    for j in range(nph5.shape[0]):
        is_eq = True
        ###check if eq
        for ii in range(nph5.shape[1]):
            for ij in range(nph5.shape[2]):
                for ik in range(nph5.shape[3]):
                    if j == 0 and not within_eps(nph5[j, ii, ij, ik], npcv[j, ii, ij, ik]) and not ik == 2 and not ik == 3:
                        print(ii, ij, ik, nph5[j, ii, ij, ik], npcv[j, ii, ij, ik])
                    elif not within_eps(nph5[j, ii, ij, ik], npcv[j, ii, ij, ik]):
                        is_eq = False
                        print(j, ii, ij, ik, nph5[j, ii, ij, ik], npcv[j, ii, ij, ik])
                        break
                if not is_eq:
                    break
            if not is_eq:
                break
        break
        if not is_eq:
            #print("unequal")
            tally += 1
        if i == 0 and j == 0:

            print("nph5")
            #for ii in range(nph5.shape[1]):
            #    for ij in range(nph5.shape[2]):
            #        for ik in range(nph5.shape[3]):
            #            print(ii, ij, ik, nph5[i, ii, ij, ik], npcv[i, ii, ij, ik])


print(tally)