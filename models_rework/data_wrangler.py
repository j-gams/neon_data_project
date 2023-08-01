### load raster data
### deal with train/test splits
### normalize??
### allow for sampling in the model

from osgeo import gdal
import numpy as np

class data_wrangler:

    def __init__ (self, rootdir, filenames, cube_res, batch_size):
        ### set up parameters etc
        self.layer_locs = []
        for fname in filenames:
            self.layer_locs.append(rootdir + fname)
        self.layer_names = []
        self.layer_nodata = []
        self.layer_size = []
        self.layer_crs = []
        self.layer_data = []
        self.cube_res = cube_res
        self.xy = [] # ids of y data
        self.mode = "train"
        self.train_fold = 0
        self.batch_size = batch_size


    def load(self):
        ### import raster layers, get data and crs info
        for item in self.layer_locs:
            self.layer_names.append(item.split("/")[-1].split(".")[0])
            layer_raster = gdal.Open(item)
            rasterband = layer_raster.GetRasterBand(1)
            self.layer_nodata.append(rasterband.GetNoDataValue())
            self.layer_size.append((layer_raster.RasterXSize, layer_raster.RasterYSize))
            tulh, tpxh, _, tulv, _, tpxv = layer_raster.GetGeoTransform()
            tpxv = abs(tpxv)
            self.layer_crs.append((tulh, tulv, tpxh, tpxv))
            self.layer_data.append(layer_raster.ReadAsArray().transpose())
            del rasterband
            del layer_raster

        ### get split info
        ### ...
        ### test_ids = ...
        ### train_folds = ...
        ### val_folds = ...


    def setup(self, test=0.3, val=0.2, folds=5, seed=100807, prenormalized=True):
        ### orchestrate normalization & everything else

        ### deal with y resolution
        ### structure of [[layer index, resolution]...]
        ### idea: we only care about the lowest resolution (highest number) to align samples with
        ### --> find the lowest res y
        y_layer_max = 0
        self.y_layer_loc = 0
        for i in range(len(self.cube_res)):
            if self.xy[i] == 1:
                if self.cube_res[i] > y_layer_max:
                    y_layer_max = self.cube_res[i]
                    self.y_layer_loc = i

        ### determine the cube size
        self.expected_cube_size = []
        for i in range(len(self.cube_res)):
            if self.cube_res[i] == y_layer_max:
                self.expected_cube_size.append(1)
            else:
                self.expected_cube_size.append(y_layer_max // self.cube_res[i] + 1)

        ### make train/test/val splits
        self.make_splits(test, val, folds, seed)

        ### normalize
        self.normalize_data(True, prenormalized)

        ### help with x, y differentiation
        xids = []
        yids = []
        for i in range(len(self.layer_data)):
            if self.xy[i] == 0:
                xids.append(i)
            else:
                yids.append(i)
        self.x_ids = np.array(xids)
        self.y_ids = np.array(yids)

    def make_coord_list(self, ids, dims):
        coord_list = [[], []]
        ### dims ... (r, c)
        ### row = id // c ... col = id % c
        for elt in ids:
            coord_list[0].append(elt // dims[1])
            coord_list[1].append(elt % dims[1])
        return np.array(coord_list)

    def make_splits(self, test=0.3, val=0.2, folds=5, seed=100807):
        ### make train/test splits
        base_shape = self.layer_data[self.y_layer_loc].shape
        total_samples = base_shape[0] * base_shape[1]
        total_indices = np.arange(total_samples)
        ### random seed
        np.random.seed(seed)
        np.random.shuffle(total_indices)
        self.test_indices = total_indices[:int(total_samples * test)]
        self.test_coords = self.make_coord_list(self.test_indices, base_shape)
        self.remaining_indices = total_indices[int(total_samples * test):]
        self.remaining_coords = self.make_coord_list(self.remaining_indices, base_shape)
        self.val_fold_indices = []
        self.val_fold_coords = []
        self.train_fold_indices = []
        self.train_fold_coords = []
        for i in range(len(folds)):
            np.random.shuffle(self.remaining_indices)
            self.val_fold_indices.append(self.remaining_indices[:int(len(self.remaining_indices)*val)])
            self.val_fold_coords.append(self.make_coord_list(self.val_fold_indices[-1], base_shape))
            self.train_fold_indices.append(self.remaining_indices[int(len(self.remaining_indices)*val):])
            self.train_fold_coords.append(self.make_coord_list(self.val_fold_indices[-1], base_shape))

        self.n_folds = folds

    def normalize_data(self, by_layer=True, prenormalized=False):
        norm_min_loc = "norm_mins.csv"
        norm_max_loc = "norm_maxs.csv"
        if prenormalized:
            ### load...
            self.smins = np.genfromtxt(norm_min_loc, delimiter=',')
            self.smaxs = np.genfromtxt(norm_max_loc, delimiter=',')
        else:
            ### do normalization
            print("computing normalization parameters...")
            self.smins, self.smaxs = self.roundup_cube_values(self.remaining_coords)
            print("applying normalization...")
            self.smins = np.array(self.smins)
            self.smaxs = np.array(self.smaxs)
            print("saving normalization parameters...")
            np.savetxt(norm_min_loc, self.smins, delimiter=",")
            np.savetxt(norm_max_loc, self.smaxs, delimiter=",")

        ### execute normalization
        for i in range(len(self.layer_data)):
            self.layer_data[i] = (self.layer_data[i]-self.smins[i]) / (self.smaxs[i] - self.smins[i])

    def shuffle(self):
        np.random.shuffle(self.meta_index)

    def set_mode(self, mode):
        ### train/combined/test
        self.data_mode = mode
        if mode == "train":
            self.meta_index = np.arange(len(self.train_fold_indices[self.train_fold]))
            self.current_ids = self.train_fold_indices[self.train_fold]
            self.current_geo = self.train_fold_coords[self.train_fold]
        elif mode == "validate":
            self.meta_index = np.arange(len(self.val_fold_indices[self.train_fold]))
            self.current_ids = self.val_fold_indices[self.train_fold]
            self.current_geo = self.val_fold_coords[self.train_fold]
        elif mode == "combined":
            self.meta_index = np.arange(len(self.remaining_indices))
            self.current_ids = self.remaining_indices
            self.current_geo = self.remaining_coords
        elif mode == "test":
            self.meta_index = np.arange(len(self.test_indices))
            self.current_ids = self.test_indices
            self.current_geo = self.test_coords
        self.lenn = int(np.ceil(self.meta_index.shape[0] / self.batch_size))
        self.shuffle()

    def set_fold(self, fold):
        ### fold number
        self.train_fold = fold

    def geo_idx(self, cx, cy, geopack): #ulh, ulv, psh, psv):
        ulh, ulv, psh, psv = geopack
        ix = (cx - ulh) / psh
        iy = (ulv - cy) / psv
        return ix, iy

    def idx_geo(self, ix, iy, geopack): #ulh, ulv, psh, psv):
        ulh, ulv, psh, psv = geopack
        cx = ulh + (ix * psh)
        cy = ulv - (iy * psv)
        return cx, cy

    def compute_offsets(self):
        self.center_offset = []
        self.half_offset = []
        for i in range(len(self.layer_data)):
            self.center_offset.append(self.expected_cube_size[i] // 2)
            self.half_offset
            if self.expected_cube_size[i] % 2 == 0:
                self.half_offset.append(0.5)
            else:
                self.half_offset.append(0)

    def roundup_cube_values(self, sampleids, debug=True):
        samples_mins = []
        samples_maxs = []
        self.compute_offsets()
        for i in range(len(self.layer_data)):
            samples_mins.append(float("inf"))
            samples_maxs.append(float("-inf"))
        for j in range(sampleids.shape[0]):
            ### convert base sampleid to geo
            geo_ctr = self.idx_geo(sampleids[0, j] + 0.5, sampleids[0, j] + 0.5,
                                   self.layer_crs[self.y_layer_loc])
            for i in range(len(self.layer_data)):
                if i == self.y_layer_loc:
                    samples_mins[i] = min(samples_mins[i], self.layer_data[i][sampleids[0, j], sampleids[1, j]])
                    samples_maxs[i] = max(samples_maxs[i], self.layer_data[i][sampleids[0, j], sampleids[1, j]])
                elif self.expected_cube_size[i] == 1:
                    tidi, tidj = self.geo_idx(geo_ctr[0], geo_ctr[1], self.layer_crs[i])
                    samples_mins[i] = min(samples_mins[i], self.layer_data[i][(int(tidi), int(tidj))])
                    samples_maxs[i] = max(samples_maxs[i], self.layer_data[i][(int(tidi), int(tidj))])
                else:
                    ### determine UL
                    ### if odd: floor((x, y)) - (oi/2, oj/2)
                    ### if even: floor((x, y) + (.5, .5)) - (oi/2, oj/2)
                    tidi, tidj = self.geo_idx(geo_ctr[0], geo_ctr[1], self.layer_crs[i])
                    sulx = int(tidi + self.half_offset[i]) - self.center_offset[i]
                    suly = int(tidj + self.half_offset[i]) - self.center_offset[i]

                    temp = self.layer_data[i][sulx:sulx+self.expected_cube_size[i],
                                                     suly:suly+self.expected_cube_size[i]].reshape(-1)



                    samples_mins[i] = min(samples_mins[i], np.nanmin(temp))
                    samples_maxs[i] = max(samples_maxs[i], np.nanmax(temp))
        return samples_mins, samples_maxs

    def get_cubes(self, batch):
        xcubes = []
        ycubes = []
        
        ### batch[0] is r coords, [1] is y coords
        geox, geoy = self.idx_geo(batch[0] + 0.5, batch[1] + 0.5,
                                  self.layer_crs[self.y_layer_loc])

        ### c = np.arange(4) + np.zeros((4,1))
        ### -> c.reshape((1, 4, 4)) + ul.reshape(3,1,1)
        ### c = ((np.arange(4) + np.zeros((4,1))).reshape((1, 4, 4)) + np.array([0, 5, 3]).reshape((3,1,1))).astype(int)
        ### r = np.transpose(ba + np.zeros((4, 1)))
        ### -> r.reshape((1, 4, 4)) + ulj.reshape(3,1,1)
        ### r = ((np.transpose(np.arange(4) + np.zeros((4, 1)))).reshape((1, 4, 4)) + np.array([1, 2, 4]).reshape((3,1,1))).astype(int)

        ### get size and bounds of layer
        for xid in self.x_ids:
            ### get center coords
            tidi, tidj = self.geo_idx(geox, geoy, self.layer_crs[xid])
            sulx = np.floor(tidi + self.half_offset[xid]) - self.center_offset[xid]
            suly = np.floor(tidj + self.half_offset[xid]) - self.center_offset[xid]
            sulx = sulx.astype(int)
            suly = suly.astype(int)
            xcubes.append(self.layer_data[((np.transpose(np.arange(self.expected_cube_size[xid]) + np.zeros((self.expected_cube_size[xid], 1)))).reshape((1, self.expected_cube_size[xid], self.expected_cube_size[xid])) + batch[0].reshape((len(batch[0]),1,1))).astype(int),
                        ((np.arange(self.expected_cube_size[xid]) + np.zeros((self.expected_cube_size[xid],1))).reshape((1, self.expected_cube_size[xid], self.expected_cube_size[xid])) + batch[1].reshape((len(batch[0]),1,1))).astype(int)])

        for yid in self.yids:
            tidi, tidj = self.geo_idx(geox, geoy, self.layer_crs[xid])
            sulx = np.floor(tidi + self.half_offset[xid]) - self.center_offset[xid]
            suly = np.floor(tidj + self.half_offset[xid]) - self.center_offset[xid]
            sulx = sulx.astype(int)
            suly = suly.astype(int)
            ycubes.append(self.layer_data[((np.transpose(np.arange(self.expected_cube_size[yid]) + np.zeros((self.expected_cube_size[yid], 1)))).reshape((1, self.expected_cube_size[yid], self.expected_cube_size[yid])) + batch[0].reshape((len(batch[0]),1,1))).astype(int),
                        ((np.arange(self.expected_cube_size[yid]) + np.zeros((self.expected_cube_size[yid],1))).reshape((1, self.expected_cube_size[yid], self.expected_cube_size[yid])) + batch[1].reshape((len(batch[0]),1,1))).astype(int)])
        return xcubes, ycubes

    def getindices(self, idx):
        return self.meta_index[idx*self.batch_size: min(((idx+1) * self.batch_size), self.meta_index.shape[0])]

    def __len__ (self):
        return self.lenn

    def __getitem__ (self, idx):
        ### [[...],
        ###  [...]]
        real_ids = self.get_cube(self.current_geo[:,self.getindices(idx)])
        return self.get_cubes(real_ids)

    def on_epoch_end(self):
        np.random.shuffle(self.meta_index)

class miniwrangler:
    def __init__(self, parent):
        self.parent = parent

    def on_epoch_end(self):
        pass

    def get_cubes(self, batch):
        pass

    