### train model, report info

### import ...
import numpy as np
import mutils
import logger
import time
import os
import pickle
from datetime import datetime

import train_1

def train(dataset, modelclass, hparams, save_name, params):
    k_folds = params["folds"]
    start_time = datetime.now()    
    #logger = 
    #performance = []
    #models = []
    ### exp_name is save_name + _ + time
    nowtime = str(start_time.now()).split(" ")
    nowtimestr = nowtime[0] + "_" + nowtime[1]
    hparams["save_location"] = save_name + "_" + nowtimestr
    mlogger = logger.mdl_log(save_name + "_" + nowtimestr, save_name, nowtimestr,
            hparams, params)
    os.system("mkdir ./saved_models/" + save_name + "_" + nowtimestr)
    for i in range(k_folds):
        print("* BEGINNING FOLD " + str(i))
        fsave = "saved_models/" + save_name + "_" + nowtimestr + "/fold_"
        fsave += str(i)
        os.system("mkdir " + fsave)
        #if train
        #if "train_realtime" in params["metrics"]:
        #    fold_start_time = time.time()
        #if "train_processtime" in params["metrics"]:
        #    fold_start_ptime = time.process_time()
        if params["mode"] == "train":
            model = modelclass(hparams, fsave)
            ### TODO - move epochs to hparams
            fold_start_time = time.time()
            fold_start_ptime = time.process_time()
            model.train(dataset.train[i], dataset.validation[i])
            fold_train_time = time.time() - fold_start_time
            fold_train_ptime = time.time() - fold_start_ptime
        elif params["mode"] == "load":
            print("not yet implemented")
        yhat = model.predict(dataset.validation[i])
        ### compute metrics
        ### ...
        computed_metrics = mutils.compute_metrics(dataset.validation[i].y, yhat,
                params["metrics"], [fold_train_time, fold_train_ptime]) 
        mlogger.add_record(computed_metrics, i)
        ### TODO -- pickle model
        print("* FOLD " + str(i) + " COMPLETED. SAVING MODEL...")
        if params["save_models"]:
            with open(fsave + "/mdl_" + save_name + ".pickle", 'wb') as dump_file:
                pickle.dump(model, dump_file)
        print("* DONE")
    ### do not run on test yet I guess?
    ### save log
    mlogger.log_record()
    ### done? I guess
    print("done training model")

