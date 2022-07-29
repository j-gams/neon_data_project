### utils
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def compute_metrics(y, yhat, metrics, times):
    #print("MUTIL METRICS:", metrics)
    metricvals = []
    for m in metrics:
        if m == "mean_squared_error":
            tval = 0
            for i in range(len(yhat)):
                tval += math.pow(y[i] - yhat[i], 2)
            tval /= len(y)
            metricvals.append(tval)
        elif m == "mean_absolute_error":
            tval = 0
            for i in range(len(yhat)):
                tval += abs(y[i] - yhat[i])
            tval /= len(y)
            metricvals.append(tval)
        elif m == "train_realtime":
            metricvals.append(times[0])
        elif m == "train_processtime":
            metricvals.append(times[1])
        else:
            metricvals.append(float("nan"))
    return metricvals

def compute_by_sample(y, yhat, metric):
    if metric == "mean_absolute_error":
        ret_err = np.abs(y-yhat)
    elif metric == "mean_squared_error":
        ret_err = (y-yhat) ** 2
    return ret_err

def np_mean(x, channel):
    ret = []
    for i in range(len(x)):
        x_i = x[i][0]
        #print(x_i)
        for j in range(x_i.shape[0]):
            ret.append(np.mean(x_i[j,:,:,channel]))
    return np.array(ret)

def np_mode(x, channel):
    ret = []
    for i in range(len(x)):
        x_i = x[i][0]
        for j in range(x_i.shape[0]):
            ret.append(stats.mode(x_i[j,:,:,channel].flatten())[0][0])
            #print(ret[-1])
    return np.array(ret)

def spec_graphs(eval_x, eval_y, yhat, channel_list, modelname, saveat):
    ### 1. mean error by ecos
    ### 2. sq error by ecos
    ### 3. sq error by elev
    ### 4. sq error by land cover
    ### 5. sq error by aspect
    ### 6. sq error by slope

    mse = compute_by_sample(eval_y, yhat, "mean_squared_error")
    mae = compute_by_sample(eval_y, yhat, "mean_absolute_error")
    cnames = ["elevation", "landcover", "slope", "aspect"]
    cfuncs = [np_mean, np_mode, np_mean, np_mean]
    ### 0
    plt.figure()
    plt.scatter(eval_y, yhat)
    plt.title("predicted values vs actual ecostress values, " + modelname)
    plt.xlabel("ecostress value")
    plt.ylabel("predicted values")
    plt.savefig(saveat + "/y_yhat.png")
    plt.cla()
    plt.close()

    ### 1
    plt.figure()
    plt.scatter(eval_y, mse)
    plt.title("Squared error over ecostress values, " + modelname)
    plt.xlabel("ecostress value")
    plt.ylabel("squared error")
    plt.savefig(saveat + "/sq_error_by_ecos.png")
    plt.cla()
    plt.close()

    ### 2
    plt.figure()
    plt.scatter(eval_y, mae)
    plt.title("Absolute errmutior over ecostress values, " + modelname)
    plt.xlabel("ecostress value")
    plt.ylabel("absolute error")
    plt.savefig(saveat + "/abs_error_by_ecos.png")
    plt.cla()
    plt.close()


    for cidx in channel_list:
        plt.figure()
        cx = cfuncs[cidx](eval_x, cidx)
        #print(cx.shape)
        #print(mse.shape)
        plt.scatter(cx, mse)#np.mean(eval_x[:,:,cidx]))
        plt.title("Squared error by average " + cnames[cidx] + ", " + modelname)
        plt.xlabel("sample average " + cnames[cidx])
        plt.ylabel("squared error")
        plt.savefig(saveat + "/sq_error_"+cnames[cidx]+".png")
        plt.cla()
        plt.close()

    
