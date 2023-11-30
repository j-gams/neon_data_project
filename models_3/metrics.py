from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
### plot - performance over folds - one axes
### compile results in table & save

### compute mse
def metric_mse(y, yhat, mode="geo", granularity=["single"]):
    # mode is either overall - average for whole sample, or each - each individual sample
    # assume y, yhat are shape (samples, i, j)
    ret = []
    if mode == "geo":
        mse = (yhat - y) ** 2
        if "single" in granularity:
            ret.append(mse.mean(axis=(1, 2)))
        if "each" in granularity:
            ret.append(mse)
        if "overall" in granularity:
            ret.append(mse.mean())
    elif mode == "flatten":
        mse = (yhat[:,:,:,0] - y) ** 2
        if "single" in granularity:
            ret.append(mse.mean(axis=(1, 2)).flatten())
        if "each" in granularity:
            ret.append(mse.flatten())
        if "overall" in granularity:
            ret.append(mse.mean())
    dimcheck = []
    for elt in ret:
        try:
            dimcheck.append(elt.shape)
        except:
            dimcheck.append("scalar")
    print("mse dimensions check", dimcheck)
    return ret

### compute mae
def metric_mae(y, yhat, mode="geo", granularity=["single"]):
    # mode is either overall - average for whole sample, or each - each individual sample
    # assume y, yhat are shape (samples, i, j)
    ret = []
    if mode == "geo":
        mae = np.abs(yhat - y)
        if "single" in granularity:
            ret.append(mae.mean(axis=(1, 2)))
        if "each" in granularity:
            ret.append(mae)
        if "overall" in granularity:
            ret.append(mae.mean())
    elif mode == "flatten":
        mae = np.abs(yhat[:,:,:,0] - y)
        if "single" in granularity:
            ret.append(mae.mean(axis=(1, 2)).flatten())
        if "each" in granularity:
            ret.append(mae.flatten())
        if "overall" in granularity:
            ret.append(mae.mean())
    dimcheck = []
    for elt in ret:
        try:
            dimcheck.append(elt.shape)
        except:
            dimcheck.append("scalar")
    print("mae dimensions check", dimcheck)
    return ret


### compute r2?
def metric_r2(y, yhat, mode="geo"):
    if mode == "geo":
        y = y.flatten()
        yhat = yhat.flatten()
    slope, intercept, r_value, p_value, std_err = stats.linregress(y, yhat)
    return r_value