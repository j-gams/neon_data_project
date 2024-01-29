### data obj

from osgeo import gdal
import numpy as np
import h5py
import tensorflow.keras.utils as kr_utils

class data_wrangler (kr_utils.Sequence):
    def __init__ (self, rootdir, n_layers, n_folds, cube_dims, batch_size, buffer_nodata, x_ids, y_ids):
        ### handle basic parameters
        self.n_layers = n_layers
        self.buffer_nodata = buffer_nodata
        self.cube_res = cube_dims
        self.x_ids = x_ids
        self.y_ids = y_ids
        self.use_y_ids = list(y_ids)
        self.batch_size = batch_size
        self.mode = "train"
        self.train_fold = 0
        self.n_folds = n_folds

        ### load h5 data
        self.layer_locs = []
        self.h5_src = []
        self.h5_data = []
        for i in range(n_layers):
            self.layer_locs.append(rootdir + "/layer_"+str(i)+".h5")
            self.h5_src.append(h5py.File(self.layer_locs[i],'r'))
            self.h5_data.append(self.h5_src[i]["data"])

        ### load index data
        self.test_index = np.genfromtxt(rootdir + "/test.csv", delimiter=',')
        self.combined_index = np.genfromtxt(rootdir + "/remaining.csv", delimiter=',')
        self.train_ids = []
        self.val_ids = []
        for i in range(n_folds):
            self.train_ids.append(np.genfromtxt(rootdir+"/train_"+str(i)+".csv", delimiter=','))
            self.val_ids.append(np.genfromtxt(rootdir+"/val_"+str(i)+".csv", delimiter=','))

        ### load normalization data
        self.combined_min = np.genfromtxt(rootdir + "/norm_layer_mins_combined.csv", delimiter=',')
        self.combined_max = np.genfromtxt(rootdir + "/norm_layer_maxs_combined.csv", delimiter=',')
        self.fold_mins = []
        self.fold_maxs = []
        for i in range(self.n_folds):
            self.fold_mins.append(np.genfromtxt(rootdir + "/norm_layer_mins_fold_"+str(i)+".csv", delimiter=','))
            self.fold_maxs.append(np.genfromtxt(rootdir + "/norm_layer_maxs_fold_"+str(i)+".csv", delimiter=','))

        self.set_mode("train")

    def set_mode(self, mode):
        self.mode = mode
        self.index_len = 0
        if self.mode == "train":
            self.index_len = self.train_ids[self.train_fold].shape[0]
        elif self.mode == "val":
            self.index_len = self.val_ids[self.train_fold].shape[0]
        else:
            self.index_len = self.combined_index.shape[0]
        self.lenn = int(np.ceil(self.index_len / self.batch_size))
        self.shuffle()

    def set_fold(self, fold):
        self.train_fold = fold

    def shuffle(self):
        if self.mode == "train":
            np.random.shuffle(self.train_ids[self.train_fold])
        elif self.mode == "val":
            np.random.shuffle(self.val_ids[self.train_fold])
        else:
            np.random.shuffle(self.combined_index)

    def getindices(self, idx):
        if self.mode == "train":
            return self.train_ids[self.train_fold][idx*self.batch_size: min(((idx+1) * self.batch_size), self.index_len)]
        elif self.mode == "val":
            return self.val_ids[self.train_fold][idx*self.batch_size: min(((idx+1) * self.batch_size), self.index_len)]
        else:
            return self.combined_index[idx*self.batch_size: min(((idx+1) * self.batch_size), self.index_len)]

    def __len__ (self):
        return self.lenn

    def apply_norm(self, npar, k):
        if self.mode == "train" or self.mode == "val":
            return (npar - self.fold_mins[self.train_fold][k]) / (self.fold_maxs[self.train_fold][k] - 
                                                               self.fold_mins[self.train_fold][k])
        else:
            return (npar - self.combined_min[k]) / (self.combined_max[k] - self.combined_min[k])

    def set_single_y(self, set_to):
        self.use_y_ids = [self.y_ids[set_to]]
    def set_multi_y(self):
        self.use_y_ids = list(self.y_ids)

    def __getitem__ (self, idx):
        ### load cubes
        ### apply normalization
        ### return
        ret_indices = np.sort(self.getindices(idx)).astype(int)
        ### ret x is formatted [layer][batch, i, j]
        ret_x = []
        ret_y = []
        for i in range(len(self.x_ids)):
            ret_x.append(np.zeros((len(ret_indices), self.cube_res[self.x_ids[i]], self.cube_res[self.x_ids[i]])))
        #for i in range(len(self.y_ids)):
        #    ret_y.append(np.zeros((len(ret_indices), self.cube_res[self.y_ids[i]], self.cube_res[self.y_ids[i]])))
        for i in range(len(self.use_y_ids)):
            ret_y.append(np.zeros((len(ret_indices), self.cube_res[self.use_y_ids[i]], self.cube_res[self.use_y_ids[i]])))
        for j in range(len(self.x_ids)):
            ret_x[j][:,:,:] = self.apply_norm(np.array(self.h5_data[self.x_ids[j]][ret_indices, :, :]), self.x_ids[j])

        #for j in range(len(self.y_ids)):
        #    ret_y[j][:,:,:] = self.apply_norm(np.array(self.h5_data[self.y_ids[j]][ret_indices, :, :]), self.y_ids[j])
        for j in range(len(self.use_y_ids)):
            ret_y[j][:,:,:] = self.apply_norm(np.array(self.h5_data[self.use_y_ids[j]][ret_indices, :, :]), self.use_y_ids[j])

        return ret_x, ret_y

    def on_epoch_end(self):
        self.shuffle()