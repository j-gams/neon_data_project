### utils
import math

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
