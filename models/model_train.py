### train model, report info

### import ...
import numpy as np
import mutils
import logger
import time
from datetime import datetime


def train(self, dataset, modelclass, hparams, save_name, params):
    k_folds = params["folds"]
    start_time = datetime.now()    
    #logger = 
    #performance = []
    #models = []
    ### exp_name is save_name + _ + time
    nowtime = str(start_time.now()).split(" ")
    nowtimestr = nowtimestr[0] + "_" + nowtimestr[1]
    mlogger = logger.mdl_log(save_name + "_" + nowtimestr, save_name, nowtimestr,
            hparams, params)
    
    for i in range(k_folds):
        #if train
        if "train_realtime" in params["metrics"]:
            fold_start_time = time.time()
        if "train_processtime" in params["metrics"]:
            fold_start_ptime = time.process_time()
        if params["mode"] == "train":
            model = modelclass(hparams)
            ### TODO - move epochs to hparams
            fold_start_time = time.time()
            fold_start_ptime = time.process_time()
            model.train(dataset.train[i], dataset.validation[i], tepochs=hparams["epochs"])
            fold_train_time = time.time() - fold_start_time
            fold_train_ptime = time.time() - fold_start_ptime
        elif params["mode"] == "load":
            print("not yet implemented")
        yhat = model.predict(datase)
        ### compute metrics
        ### ...
        computed_metrics = mutils.compute_metrics(dataset.validation[i].y, yhat,
                params["metrics"], [fold_train_time, fold_train_ptime]) 
        mlogger.add_record(computed_metrics, i)
        ### TODO -- pickle model
    ### do not run on test yet I guess?
    ### save log
    mlogger.log_record()
    ### done? I guess
    print("done training model")

