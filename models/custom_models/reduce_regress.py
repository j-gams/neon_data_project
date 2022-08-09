### PCA

import numpy as np

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.decomposition import PCA

class decompose_regress:

    def __init__ (self, hparam_dict, save_dir):
        self.crdict = dict()
        self.verbosity = 2
        self.rmode = "lr"
        self.dmode = "pca"
        self.dropmode = "none"
        self.dropout = []
        self.dmodel = None
        self.rmodel = None
        self.reduceto = 5
        self.modelname = ""
        self.avg_channel = True        
        for key in hparam_dict:
            if key == "model_name":
                self.modelname = hparam_dict[key]
            if key == "save_location":
                self.saveloc = hparam_dict[key]
            if key == "reduce_to":
                self.reduceto = hparam_dict[key]
            if key == "rmode":
                self.rmode = hparam_dict[key]
            if key == "dmode":
                self.dmode = hparam_dict[key]
            if key == "dropout":
                self.dropmode = hparam_dict[key]["mode"]
                self.dropout = hparam_dict[key]["channels"]


        #if self.dropmode == "keep":
        #    self.keeplen = len(self.dropout)
        #elif# self.dropmode == "drop":
        #    self.keeplen = self.imgsize[2] - len(self.dropout)
        #else:
        #    self.keeplen = self.imgsize[2]
        
        #if self.dmode == "pca":
        #    self.dmodel = PCA(n_components=self.reduceto)
        #elif self.dmode == "svd":
        #    pass
        #if self.rmode == "lr":
        #    self.rmodel = LinearRegression()

    def train(self, train_data, validation_data):
        self.data_channels = train_data.nchannels
        if self.dropmode == "keep":
            self.keeplen = len(self.dropout)
        elif self.dropmode == "drop":
            self.keeplen = self.data_channels - len(self.dropout)
        else:
            self.keeplen = self.data_channels
        self.change_restore(train_data, "c", "train_ae")
        self.reduceto = min(self.reduceto, self.keeplen)
        #if self.dropmode == "drop":
        #    self.keeplen = train_data.nchannels
        #elif self.dropmode == "none":
        #    self.keeplen = train_data.dims[2]
        if self.dmode == "pca":
            self.dmodel = PCA(n_components=self.reduceto)
        elif self.dmode == "svd":
            pass
        if self.rmode == "lr":
            self.rmodel = LinearRegression()
        
        keep_mu = 1
        if not self.avg_channel:
            keep_mu = train_data.dims[0] * train_data.dims[1]
        
        fulltrain = np.zeros((train_data.get_n_samples(), self.keeplen*keep_mu))
        for i in range(len(train_data)):
            fulltrain[i*train_data.batch_size:min(len(fulltrain),
                                                      (i+1)*train_data.batch_size),
                          :] = self.dtransform(train_data[i])
        print("train dims:", fulltrain.shape, self.keeplen)
        print("keepids:", train_data.keep_ids)
        self.dmodel.fit(fulltrain)
        
        transform_train = self.dmodel.transform(fulltrain)
        self.rmodel.fit(transform_train, train_data.y)
        self.change_restore(train_data, "r", "train_ae")

    def predict(self, x_predict):
        dumb_out = []
        self.change_restore(x_predict, "c", "eval")
        for i in range(len(x_predict)):
            tpred = self.dmodel.transform(self.dtransform(x_predict[i]))
            dumb_out.append(self.rmodel.predict(tpred))
        ret_y = np.array(dumb_out).reshape(-1).flatten()
        self.change_restore(x_predict, "r", "eval")
        return ret_y
    
    def dtransform(self, data):
        if self.avg_channel:
            return self.mean_itr(data, nchannels = self.keeplen)
        else:
            return data
    
    def mean_itr(self, data, nchannels):
        #print(type(data))
        #print("mean itr", data.dims)
        ret_vals = np.zeros((data.shape[0], nchannels))
        n_in_c = data.shape[1]//nchannels
        for i in range(nchannels):
            ret_vals[:,i] = np.mean(data[:, [(ii*nchannels) + i for ii in range(n_in_c)]], axis=1)
        return ret_vals

    def change_restore(self, data, c_r, name):
        if c_r == "c":
            ### autoencoder mode
            self.crdict[name] = [data.return_format,
                                     data.flat_mode,
                                     data.keep_ids,
                                     data.drop_channels]
            data.set_return("x")
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
           data.set_return(self.crdict[name][0])
           data.set_flatten(self.crdict[name][1])
           data.set_keeps(self.crdict[name][2])
           data.set_drops(self.crdict[name][3])
    
    
