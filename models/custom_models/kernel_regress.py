### Written by Jerry Gammie @j-gams

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel, sigmoid_kernel
import mutils

class kernel_regress:
    def __init__ (self, hparam_dict, save_dir):
        self.verbosity = 2
        self.dropout = []
        self.dropmode = "none"
        self.keeplen = 0

        self.crdict = dict()
        self.avg_channel = True
        self.self_norm = False
        kname = "rbf"
        self.avg_channel=True

        for key in hparam_dict:
            if key == "model_name":
                self.modelname = hparam_dict[key]
            elif key == "alpha":
                self.alpha = hparam_dict[key]
            elif key == "kernel":
                kname = hparam_dict[key]
            elif key == "dropout":
                self.dropmode = hparam_dict[key]["mode"]
                self.dropout = hparam_dict[key]["channels"]
                if self.dropmode == "keep":
                    self.keeplen = len(self.dropout)
            elif key == "avg_channel":
                self.avg_channel = hparam_dict[key]
            elif key == "normalize":
                self.self_norm = hparam_dict[key]
            elif key == "verbosity":
                self.verbosity = hparam_dict[key]

        if kname == "rbf":
            self.kernel = "rbf"
        elif kname == "lin":
            self.kernel = "linear"
        elif kname == "ply":
            self.kernel = "polynomial"
        elif kname == "sig":
            self.kernel = "sigmoid"
            
        ### setup
        self.model = KernelRidge(alpha = self.alpha, kernel=self.kernel)
            
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
        self.data_channels = train_data.nchannels
        if self.dropmode == "keep":
            self.keeplen = len(self.dropout)
        elif self.dropmode == "drop":
            self.keeplen = self.data_channels - len(self.dropout)
        else:
            self.keeplen = self.data_channels
        """ if self.dropmode == "drop":
            self.keeplen = train_data.nchannels - len(self.dropout)
        elif self.dropmode == "none":
            self.keeplen = train_data.nchannels"""
        keep_mu = 1
        if not self.avg_channel:
            keep_mu = train_data.dims[0] * train_data.dims[1]

        self.change_restore(train_data, "c", "train")

        fulltrain = np.zeros((train_data.get_n_samples(), self.keeplen*keep_mu))
        for i in range(len(train_data)):
            fulltrain[i*train_data.batch_size:min(len(fulltrain),
                                                      (i+1)*train_data.batch_size),
                          :] = self.dtransform(train_data[i][0])
        print(fulltrain.shape)
        print(fulltrain[0].shape)
        self.model.fit(fulltrain, train_data.y)
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
