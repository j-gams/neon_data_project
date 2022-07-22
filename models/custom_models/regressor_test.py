### TEST REGRESSOR 1

### TEMPLATE???
### imports
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
import mutils

class test_regress:
    def __init__ (self, hparam_dict, save_dir, testmode=True, Transformer=True):
        init_count = 0
        self.verbosity = 2
        train_metric = "mean_squared_error"
        self.dropout = []
        self.dropmode = "drop"
        self.keeplen = 2
        self.crdict = dict()
        self.retain_average = False
        for key in hparam_dict:
            if key == "model_name":
                self.modelname = hparam_dict[key]
                init_count += 1
            #elif key == "save_location":
            #    self.saveloc = hparam_dict[key]
            #    init_count += 1
            elif key == "penalty":
                self.penalty = hparam_dict[key]
                init_count += 1
            elif key == "alpha":
                self.alpha = hparam_dict[key]
                init_count += 1
            #elif key == "input_size":
            #    self.imgsize = hparam_dict[key]
            #    init_count += 1
            elif key == "save_checkpoints":
                self.savechecks = hparam_dict[key]
                init_count += 1
            elif key == "use_best":
                self.usebest = hparam_dict[key]
                init_count += 1
            elif key == "save_last_epoch":
                self.savelast = hparam_dict[key]
                init_count += 1
            elif key == "train_metric":
                train_metric = hparam_dict[key]
                init_count += 1
            elif key == "epochs":
                self.n_epochs = hparam_dict[key]
                init_count += 1
            elif key == "dropout":
                self.dropmode = hparam_dict[key]["mode"]
                self.dropout = hparam_dict[key]["channels"]
                if self.dropmode == "keep":
                    self.keeplen = len(self.dropout)
            elif key == "avg_channel":
                pass
            elif key == "retain_avg":
                self.retain_avg = hparam_dict[key]
            elif key == "verbosity":
                self.verbosity = hparam_dict[key]
        if init_count != 8:
            ### did not initialize ok
            print("model not initialized correctly!")

        #self.metricset = ["mean_squared_error", "mean_absolute_error", "elasticnet"]
        self.metricset = {"mean_squared_error": "squared_error",
                     "huber": "huber",
                     "epsilon_intensive": "epsilon_intensive",
                     "squared_epsilon_intensive": "squared_epsilon_intensive"}
        if train_metric not in self.metricset:
            self.tmetric = "mean_squared_error"
        else:
            self.tmetric = train_metric
        self.model = SGDRegressor(max_iter = 50000)
        #self.model = SGDRegressor(loss=self.metricset[self.tmetric], penalty=self.penalty,
        #        alpha=self.alpha, max_iter=1)

    def mean_itr(self, data, nchannels):
        #print(type(data))
        ret_vals = np.zeros((data.shape[0], nchannels))
        n_in_c = data.shape[1]//nchannels
        for i in range(nchannels):
            ret_vals[:,i] = np.mean(data[:, [(ii*nchannels) + i for ii in range(n_in_c)]], axis=1)
        return ret_vals
    
    def change_restore(self, data, c_r, name):
        if c_r == "c":
            self.crdict[name] = [data.return_format,
                                 data.flat_mode,
                                 data.keep_ids,
                                 data.drop_channels]
            data.set_return("x")
            data.set_flatten(True)
            if self.dropmode == "keep":
                data.set_keeps(self.dropout)
                data.set_drops(data.keeps_to_drops())
            else:
                data.set_drops(self.dropout)
                data.set_keeps(data.drops_to_keeps())
        else:
           data.set_return(self.crdict[name][0])
           data.set_flatten(self.crdict[name][1])
           data.set_keeps(self.crdict[name][2])
           data.set_drops(self.crdict[name][3])

    def train(self, train_data, validation_data):
        #og_return = train_data.return_format
        #og_flatmode = train_data.flat_mode
        #og_keep_ids = train_data.keep_ids
        #og_drop_ids = train_data.drop_channels
        #train_data.set_flatten(True)
        if self.dropmode == "drop":
            self.keeplen = train_data.nchannels
        mval = 0
        self.change_restore(train_data, "c", "train")
        self.train_y = train_data.y
        #if self.dropmode == "keep":
        #    train_data.set_keeps(self.dropout)
        #    train_data.set_drops(train_data.keeps_to_drops())
        #else:
        #    train_data.set_drops(self.dropout)
        #    train_data.set_keeps(train_data.drops_to_keeps())
        #print(len(train_data))
        #train_data.set_return("x")
        #print("train")
        do_fulltrain = True
        if do_fulltrain:
            #print(train_data.get_n_samples())
            fulltrain = np.zeros((train_data.get_n_samples(), self.keeplen))
            #print(fulltrain.shape)
        for i in range(len(train_data)):
            td = self.mean_itr(train_data[i], self.keeplen)
            if do_fulltrain:
                fulltrain[i*train_data.batch_size:min(len(fulltrain), (i+1)*train_data.batch_size),:] = td
            tyd = train_data.y[train_data.getindices(i)] 
            #for j in range(len(td)):
            #    print(td[j], " > ", tyd[j])
        #print("validation")
        self.change_restore(validation_data, "c", "val")
        #for i in range(len(validation_data)):
        #    td = self.mean_itr(validation_data[i], self.keeplen)
        #    tyd = validation_data.y[validation_data.getindices(i)]
        #    for j in range(len(td)):
        #        print(td[j], " > ", tyd[j])
        self.change_restore(validation_data, "r", "val")
        #y_train_return = train_data.set_return("y")
        #train_data.set_return("x")
        if do_fulltrain:
            self.model = LinearRegression().fit(fulltrain, train_data.y)
            #self.model.fit(fulltrain, train_data.y)#LinearRegression().fit(fulltrain, train_data.y)
            return
        for j in range(self.n_epochs):
            print("epoch " + str(j) + " >", end="")
            #y_train_retiurn = train_data.set_return("y")
            #train_data.set_return("x")
            for i in range(len(train_data)):
                #if j == 0:
                #    print(train_data[i].shape)
                #    print(self.mean_itr(train_data[i], 4).shape)
                #    print(self.mean_itr(train_data[i], 4))
                #    print(train_data[i][0])
                #    #for k in range(train_data[i].shape[1]):
                #    #    print("*", train_data[i][0, k])
                #    #print(train_data.y[train_data.getindices(i)].shape)
                xtrain = train_data[i]
                if j == 0:
                    tmax = np.max(xtrain)
                    if tmax > mval:
                        mval = tmax
                train_data.set_return("y")
                ytrain = train_data[i]
                train_data.set_return("x")
                self.model.partial_fit(self.mean_itr(xtrain, self.keeplen), ytrain)
                #self.model.partial_fit(train_data[i], train_data.y[train_data.getindices(i)])
                #print("-", end="", flush=True)
            print("> done!")

            ### evaluate
            ### TODO -- save best etc.
            #print(train_data.X_ref[train_data.getindices(0)])
            #print(self.mean_itr(train_data[0], self.keeplen))
            #print(train_data.y[train_data.getindices(0)])
            if j == 0:
                print("max value encountered in training:", mval)
            #print(self.mean_itr(validation_data[0], 4))
            yhat = self.predict(validation_data, mode1="fake")
            m = mutils.compute_metrics(validation_data.y, yhat, [self.tmetric], [0, 0])[0]
            print("validation " + self.tmetric + ": " + str(m))            
            train_data.on_epoch_end()
        #self.model.fit(train_data.set_return("x"), train_data.set_return("y"))
        self.change_restore(train_data, "r", "train")
        #train_data.set_return(og_return)
        #train_data.set_flatten(og_flatmode)
        #train_data.set_drops(og_drop_ids)
        #train_data.set_keeps(og_keep_ids)
   

    def predict(self, x_predict, typein="simg", mode1="real"):
        if typein == "simg":
            dumb_out = []
            self.change_restore(x_predict, "c", "eval")
            #og_ret = x_predict.return_format
            #og_flt = x_predict.flat_mode
            #og_drp = x_predict.drop_channels
            #og_kep = x_predict.keep_ids
            #x_predict.set_return("x")
            #x_predict.set_flatten(True)
            #if self.dropmode == "keep":
            #    x_predict.set_keeps(self.dropout)
            #    x_predict.set_drops(x_predict.keeps_to_drops())
            #else:
            #    x_predict.set_drops(self.dropout)
            #    x_predict.set_keeps(x_predict.drops_to_keeps())
            #print(self.mean_itr(x_predict[0], self.keeplen))
            #print(x_predict.y[x_predict.getindices(0)])
            for i in range(len(x_predict)):
                dumb_out.append(self.model.predict(self.mean_itr(x_predict[i], self.keeplen)))
            #print(x_predict.y)
            ret_y = np.array(dumb_out).reshape(-1).flatten()
            #if mode1 != "fake":
            #    print(self.train_y)
            #    print(x_predict.y)
            #    print(ret_y)
            #    for i in range(len(x_predict)):
            #        td = self.mean_itr(x_predict[i], self.keeplen)
            #        tyd = x_predict.y[x_predict.getindices(i)]
            #        for j in range(len(td)):
            #            print(td[j], " > ", tyd[j])
            #self.model(x_predict)
            self.change_restore(x_predict, "r", "eval")
            #x_predict.set_return(og_ret)
            #x_predict.set_flatten(og_flt)
        return ret_y

        #return self.model(x_predict)
