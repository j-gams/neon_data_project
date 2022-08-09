### TEST REGRESSOR 1

### TEMPLATE???
### imports
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
import mutils

class test_regress:
    def __init__ (self, hparam_dict, save_dir, testmode=True, Transformer=True):
        self.verbosity = 2
        train_metric = "mean_squared_error"
        self.dropout = []
        self.dropmode = "none"
        self.keeplen = 0
        self.crdict = dict()
        self.avg_channel = True
        self.retain_average = False
        self.batchr = True
        self.self_norm = False
        for key in hparam_dict:
            if key == "model_name":
                self.modelname = hparam_dict[key]
            elif key == "save_location":
                ### save to memory anyhow
                pass
            elif key == "save_checkpoints":
                self.savechecks = hparam_dict[key]
            elif key == "train_metric":
                train_metric = hparam_dict[key]
            elif key == "epochs":
                self.n_epochs = hparam_dict[key]
            elif key == "use_best":
                self.usebest = hparam_dict[key]
            elif key == "save_last_epoch":
                self.savelast = hparam_dict[key]
            elif key == "penalty":
                self.penalty = hparam_dict[key]
            elif key == "alpha":
                self.alpha = hparam_dict[key]
            elif key == "dropout":
                self.dropmode = hparam_dict[key]["mode"]
                self.dropout = hparam_dict[key]["channels"]
                if self.dropmode == "keep":
                    self.keeplen = len(self.dropout)
            elif key == "avg_channel":
                self.avg_channel = hparam_dict[key]
            elif key == "retain_avg":
                self.retain_avg = hparam_dict[key]
            elif key == "batch_regress":
                self.batchr = hparam_dict[key]
            elif key == "normalize":
                self.self_norm = hparam_dict[key]
            elif key == "verbosity":
                self.verbosity = hparam_dict[key]

        if not self.batchr:
            self.retain_avg = True
        #if init_count != 8:
        #    print("model not initialized correctly!")

        self.metricset = {"mean_squared_error": "squared_error",
                     "huber": "huber",
                     "epsilon_intensive": "epsilon_intensive",
                     "squared_epsilon_intensive": "squared_epsilon_intensive"}
        if train_metric not in self.metricset:
            self.tmetric = "mean_squared_error"
        else:
            self.tmetric = train_metric
        if self.batchr:
            self.model = SGDRegressor(alpha=self.alpha, max_iter=50000)
        else:
            self.model = LinearRegression()

    def dtransform(self, data):
        if self.avg_channel:
            return self.mean_itr(data, nchannels = self.keeplen)
        else:
            return data
    
    def mean_itr(self, data, nchannels):
        #print(type(data))
        ret_vals = np.zeros((data.shape[0], nchannels))
        n_in_c = data.shape[1]//nchannels
        for i in range(nchannels):
            ret_vals[:,i] = np.mean(data[:, [(ii*nchannels) + i for ii in range(n_in_c)]], axis=1)
        return ret_vals
    
    def change_restore(self, data, c_r, name):
        if c_r == "c":
            self.crdict[name] = [data.flat_mode,
                                 data.keep_ids,
                                 data.drop_channels]
            #data.set_return("x")
            data.set_flatten(True)
            if self.dropmode == "keep":
                data.set_keeps(self.dropout)
                data.set_drops(data.keeps_to_drops())
            elif self.dropmode == "drop":
                data.set_drops(self.dropout)
                keepsl = []
                for i in range(data.nchannels):
                    if i not in self.dropout:
                        keepsl.append(i)
                data.set_keeps(keepsl)
            else:
                data.set_drops([])
                data.set_keeps(data.drops_to_keeps())
        else:
           #data.set_return(self.crdict[name][0])
           data.set_flatten(self.crdict[name][0])
           data.set_keeps(self.crdict[name][1])
           data.set_drops(self.crdict[name][2])

    def train(self, train_data, validation_data):
        ### handle keeplen... jus
        if self.dropmode == "drop":
            self.keeplen = train_data.nchannels - len(self.dropout)
        elif self.dropmode == "keep":
            self.keeplen = len(self.dropout)
        else:
            self.keeplen = train_data.nchannels
        #mval = 0

        if self.usebest:
            best_metric = float("inf")
            best_model = None
        ### change dataset settings
        self.change_restore(train_data, "c", "train")
        self.train_y = train_data.y
        ### whether to compute channel avgs and save smaller data
        if self.retain_avg and self.avg_channel:
            fulltrain = np.zeros((train_data.get_n_samples(), self.keeplen))
        elif self.retain_avg:
            fulltrain = np.zeros((train_data.get_n_samples(), self.keeplen * train_data.dims[0] * 2))
        ### do overhead
        if self.retain_avg:
            for i in range(len(train_data)):
                fulltrain[i*train_data.batch_size:min(len(fulltrain),
                                                      (i+1)*train_data.batch_size),
                          :] = self.dtransform(train_data[i][0])
        elif self.self_norm:
            for i in range(len(train_data)):
                ### do norm... not yet implemented
                pass
        
        if self.retain_avg:
            print("retain-avg fitting ... dims =", fulltrain.shape)
            self.model.fit(fulltrain, train_data.y)
            self.change_restore(train_data, "r", "train")
            
        else:
            for j in range(self.n_epochs):
                print("epoch " + str(j) + " >", end="")
                for i in range(len(train_data)):
                    xtrain, ytrain = train_data[i]
                    if j == 0:
                        tmax = np.max(xtrain)
                        if tmax > mval:
                            mval = tmax
                    self.model.partial_fit(self.dtransform(xtrain), ytrain)
                print("> done!")
                if j == 0:
                    print("max value encountered in training:", mval)
                if self.savechecks:
                    yhat = self.predict(validation_data, mode1="fake")
                    m = mutils.compute_metrics(validation_data.y, yhat, [self.tmetric], [0, 0])[0]
                    if m < best_metric:
                        best_model = self.model
                        best_metric = m
                    print("validation " + self.tmetric + ": " + str(m))            
                train_data.on_epoch_end()
            if self.usebest:
                if self.savelast:
                    self.last_model = self.model
                self.model = best_model
            self.change_restore(train_data, "r", "train")
   

    def predict(self, x_predict, typein="simg", mode1="real"):
        if typein == "simg":
            dumb_out = []
            self.change_restore(x_predict, "c", "eval")
            for i in range(len(x_predict)):
                dumb_out.append(self.model.predict(self.dtransform(x_predict[i][0])))
            ret_y = np.array(dumb_out).reshape(-1).flatten()
            self.change_restore(x_predict, "r", "eval")
        return ret_y

        #return self.model(x_predict)
