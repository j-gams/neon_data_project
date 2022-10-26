### Written by Jerry Gammie @j-gams

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import mutils

class gbregressor:
    def __init__ (self, hparam_dict, save_dir):
        self.verbosity = 2
        self.dropout = []
        self.dropmode = "none"
        self.keeplen = 0
        self.crdict = dict()
        self.transformer = "mean_itr"
        self.avg_channel = True

        self.n_ests = 100
        self.maxdepth = 3
        self.learning_rate = 0.1

        for key in hparam_dict:
            if key == "verbosity":
                self.verbosity = hparam_dict[key]
            elif key == "model_name":
                self.modelname = hparam_dict[key]
            elif key == "dropout":
                self.dropmode = hparam_dict[key]["mode"]
                self.dropout = hparam_dict[key]["channels"]
            elif key == "avg_channel":
                self.avg_channel = hparam_dict[key]
            elif key == "n_estimators":
                self.n_ests = hparam_dict[key]
            elif key == "max_depth":
                self.maxdepth = hparam_dict[key]
            elif key == "learning_rate":
                self.learning_rate = hparam_dict[key]

        self.model = GradientBoostingRegressor(n_estimators=self.n_ests, max_depth=self.maxdepth,
                                               learning_rate=self.learning_rate)

    def qprint(self, item, prio, ef = False):
        if prio < self.verbosity:
            if ef:
                print(item, end="", flush=True)
            else:
                print(item)

    def dtransform (self, data):
        if self.avg_channel:
            if self.transformer == "mean_itr":
                return self.mean_itr(data, nchannels = self.keeplen)
        else:
            return data

    def mean_itr (self, data, nchannels):
        ret_vals = np.zeros((data.shape[0], nchannels))
        n_in_c = data.shape[1] // nchannels
        for i in range(nchannels):
            ret_vals[:, i] = np.mean(data[:, [(ii * nchannels) + i for ii in range(n_in_c)]], axis=1)
        return ret_vals

    def drop_set (self, dchannels):
        if self.dropmode == "keep":
            self.keeplen = len(self.dropout)
        elif self.dropmode == "drop":
            self.keeplen = dchannels - len(self.dropout)
        else:
            self.keeplen = dchannels

    def change_restore(self, data, c_r, name):
        if c_r == "c":
            self.crdict[name] = [data.flat_mode,
                                 data.keep_ids,
                                 data.drop_channels]
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
           data.set_flatten(self.crdict[name][0])
           data.set_keeps(self.crdict[name][1])
           data.set_drops(self.crdict[name][2])

    def aggregate (self, data):
        self.change_restore(data, "c", "agg")
        keep_mu = 1
        y_agg = np.zeros(data.get_n_samples())
        if not self.avg_channel:
            keep_mu = data.dims[0] * data.dims[1]
        agg = np.zeros((data.get_n_samples(), self.keeplen*keep_mu))
        for i in range(len(data)):
            tx, ty = data[i]
            agg[i*data.batch_size : min(len(agg), (i+1)*data.batch_size), :] = self.dtransform(tx)
            y_agg[i*data.batch_size : min(len(agg), (i+1)*data.batch_size)] = ty
        print(agg.shape)
        self.change_restore(data, "r", "agg")
        return agg, y_agg

    def train(self, train_data, validation_data):
        self.drop_set(train_data.nchannels)
        npx, npy = self.aggregate(train_data)
        self.model.fit(npx, npy)

    def predict (self, x_predict, typein = "simg"):
        if typein == "simg":
            npx, _ = self.aggregate(x_predict)
            return self.model.predict(npx)
        elif typein == "arr":
            return self.model.predict(self.dtransform(x_predict))