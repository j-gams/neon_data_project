
import pandas as pd
import numpy as np
from datacube_set import satimg_set
import rfdata_loader
class datacube_loader:

    ### PARAMETERS OVERVIEW
    ### self ...
    ### expect_folds    number of cross-validation folds
    ### shuffle         whether to shuffle each epoch
    ### batch           batch sizes for each set
    ### x_ref_idx       index referencing file names for each data cube
    ### y_col_idx       index for y values in data
    ### musigs          mean, stdev to use
    ### mem             memory sensitive mode
    ### omode           per-channel or global means, stds
    ### cmode           whether to run in channel-first or channel-last mode
    ### order for thruple params is train, val, test...

    ### satimg_set(data_in, shuffle, path_prefix, batch_size, x_ref_idx, y_col_idx, mean_stds, depth_ax, dataname,
    ### mem_sensitive, observe_mode)
    ### TODO -- incorporate channel names ... load from meta/channel_names
    def __init__ (self, dataname, dataext, shuffle, batch, x_ref_idx, y_col_idx, musigs, mem, omode, cmode):
        print("building sets...")
        self.dataset_name = dataname
        #self.k_folds = expect_folds
        
        meta_folds = 0 

        ### load in
        with open("../data/" + dataname + "/fold_data/" + dataext + "/meta.txt") as fff:
            alllines = fff.readlines()
        for fline in alllines:
            metaparams = fline.split(": ")
            if metaparams[0] == "folds":
                meta_folds = int(metaparams[1])

        print("meta folds: ", meta_folds)
        self.k_folds = meta_folds
        ### load channel names

        channel_names = rfdata_loader.d1loader("../data/" + dataname + 
                "/meta/channel_names.txt")
        print("channel names: ", channel_names)
        #if musigs = "default":
        #    musigs = 
        self.channel_mode = cmode
        alldata_np = pd.read_csv("../data/" + dataname + "/datasrc/ydata.csv").to_numpy()
        self.all = alldata_np
        test_info = np.genfromtxt("../data/" + dataname + "/fold_data/" + dataext + "/test/test_set.csv", delimiter=',')
        test_info = test_info.astype(int)
        self.test_data_raw = alldata_np[test_info]
        self.validation_data_raw = []
        self.train_data_raw = []
        self.test = satimg_set(self.test_data_raw, shuffle[2], "../data/" + dataname, batch[2], x_ref_idx, y_col_idx,
                musigs[2], channel_names, dataname="test set", mem_sensitive=mem[2], observe_mode=omode[2],
                orientation = self.channel_mode)
        self.train = []
        self.validation = []
        self.train_m_s = []
        for i in range(self.k_folds):
            tval = np.genfromtxt("../data/" + dataname + "/fold_data/" + dataext + "/train_fold_"+str(i)+"/val_fold.csv",
                    delimiter=',').astype(int)
            ttrain = np.genfromtxt("../data/" + dataname + "/fold_data/" + dataext + "/train_fold_"+str(i)+"/train_fold.csv",
                    delimiter=',').astype(int)
            self.validation_data_raw.append(alldata_np[tval])
            self.train_data_raw.append(alldata_np[ttrain])
            self.train.append(satimg_set(self.train_data_raw[-1], shuffle[0], "../data/" + dataname, batch[0],
                x_ref_idx, y_col_idx, musigs[0], channel_names, dataname = "train set " + str(i), mem_sensitive=mem[0],
                observe_mode=omode[0], orientation = self.channel_mode))
            fold_m_s = self.train[-1].get_or_compute_m_s(mode_in=omode[0])
            self.train_m_s.append(fold_m_s)
            self.train[-1].apply_observed_m_s()
            self.validation.append(satimg_set(self.validation_data_raw[-1], shuffle[1], "../data/" + dataname,
                batch[1], x_ref_idx, y_col_idx, fold_m_s, channel_names, dataname = "validation set " + str(i),
                mem_sensitive = mem[1], observe_mode=omode[1], orientation = self.channel_mode))
        #test_data_in_np = pd.read_csv("../data/" + dataname + "/datasrc/fold_data/" + data_ext + "/test/test_set.csv")
        #test_data_in_np = test_data_in_np.to_numpy()
        #self.test_set = satimg_set(test_data_in_np, )

    def summarize(self):
        print("dataset overview:")
        print("  source data: " + str(self.all.shape[0]) + " samples")
        print("  test set: " + str(self.test.full_data.shape[0]) + " samples")
        print("  cross-validation folds: " + str(len(self.train)))
        for i in range(self.k_folds):
            print("  train fold " + str(i) + ": " + str(self.train[i].full_data.shape[0]) + " samples")
            print("  vali. fold " + str(i) + ": " + str(self.validation[i].full_data.shape[0]) + " samples") 
