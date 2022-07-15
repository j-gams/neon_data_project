import pandas as pd
import numpy as np
import tensorflow.keras.utils as kr_utils
import math

### TODO -- need to go through this carefully to double check certain aspects of computing means and stds, etc.
### TODO -- comment
class satimg_set (kr_utils.Sequence):
    def __init__ (self, data_in, shuffle, path_prefix, batch_size, x_ref_idx, y_col_idx,
                  mean_stds, rformat="both", orientation = "hwc", dataname = "", mem_sensitive=True, observe_mode="per"):
        print("initializing datafold: " + dataname)

        ### parameters
        ### name of data fold
        self.dataname = dataname
        ### whether to shuffle the data after each epoch
        self.shuffle = shuffle
        ### batch size
        self.batch_size = batch_size
        ### path prefix to use when loading x data cubes... essentially just file path to dataset
        self.path_prefix = path_prefix
        ### TODO -- I forget... think this is df with file locations
        self.full_data = data_in
        # expect [(n_channels,), (n_channels,)]
        ### precomputed means, std deviations
        self.mean_stds = mean_stds
        ### expect an image of this resolution
        #self.img_size = expect_img_size

        ### channel, height, width or height, width, channel
        self.ori = orientation
        
        ### whether to return x, y or just x/y
        ### so both/x/y
        self.return_format = rformat
        self.predict_mode = False
        ### split large input into constituent components
        ### legacy from "decision: trees" project
        ### TODO -- refactor to just chop out the img references and y, keep the other data in consolidated array
        ### Idea - return info, column labels...
        x_array_cols = []
        for i in range(data_in.shape[1]):
            if i != x_ref_idx and i != y_col_idx:
                x_array_cols.append(i)
        self.X = data_in[:, np.array(x_array_cols)]
        self.y = data_in[:, y_col_idx].astype('float64').flatten()
        self.X_ref = data_in[:, x_ref_idx].flatten()

        ### practice loading in one sample to get dimension info
        info_cube = self.load_dcube(self.path_prefix + self.X_ref[0], skip=True)
        ### shape to expect for all X samples
        ### example: data cubes created by match_create_set are (channels, rows, cols)
        ### or (~60, 14, 14)
        self.dims = info_cube.shape
        if self.ori == "hwc":
            self.nchannels = self.dims[2]
        else:
            self.nchannels = self.dims[0]

        if mean_stds == "default":
            mean_stds = self.blank_fake_ms(self.dims[0])
        print("datacube dimensions: ", self.dims)
        ### clean up memory
        del info_cube

        ### index the data -- useful for shuffling, generalization
        if self.shuffle:
            self.indexes = np.random.permutation(self.full_data.shape[0])
        else:
            self.indexes = np.arange(self.full_data.shape[0])

        ### shuffle the data if shuffle is True
        self.on_epoch_end()

        ### length (number of samples) within each batch
        self.lenn = int(np.ceil(self.full_data.shape[0] / self.batch_size))

        ### have we computed means and standard deviations yet?
        self.m_s_computed = False

        ### whether we need to worry about limited ram, or alternatively can we load every data cube into memory at once
        self.mem_sensitive = mem_sensitive

        ### observed means and standard deviations of the data
        """self.observed_x = [0, 0, 0]
        self.observed_x_x = [0, 0, 0]
        self.observed_mean = [0, 0, 0]
        self.observed_std = [1, 1, 1]"""
        ### REFACTOR TO HANDLE ARBITRARY SAMPLE DEPTH... DONE
        self.observed_x = [0 for ii in range(self.nchannels)]
        self.observed_x_x = [0 for ii in range(self.nchannels)]
        self.observed_mean = [0 for ii in range(self.nchannels)]
        self.observed_std = [1 for ii in range(self.nchannels)]

        ### if we aren't concerned about memory, load the entire data fold into memory
        ### BONUS: compute means and std deviations along the way
        if not mem_sensitive:
            ### preload all imgs -- only if memory is not a concern
            ### make a big np array to store all of these data cubes
            self.img_memory = np.zeros((self.full_data.shape[0], self.dims[0], self.dims[1], self.dims[2]))
            print("preloading images...")
            if observe_mode == "per" or observe_mode == "global":
                ### make np arrays for doing math
                sum_x = np.zeros(self.dims)
                sum_x_x = np.zeros(self.dims)

                ### iterate through each sample and load the data cube
                for i in range(self.full_data.shape[0]):
                    self.img_memory[i] = self.load_dcube(self.path_prefix + self.X_ref[i])
                    # compute on here
                    # temp_img = self.load_img(self.path_prefix + self.X_img_ref[i, 0], fake_ms)
                    ### do some math... needed to compute means, stds
                    sum_x += self.img_memory[i]
                    sum_x_x += self.img_memory[i] * self.img_memory[i]
                print("loaded", self.full_data.shape[0], "images.")
                ### only this far...
                ### do more setup for math... copy observed means, stds, etc over to permanent locations
                for i in range(self.nchannels):
                    if self.ori == "hwc":
                        self.observed_x[i] = np.sum(sum_x[:,:,i])
                        self.observed_x_x[i] = np.sum(sum_x_x[:,:,i])
                    else:
                        self.observed_x[i] = np.sum(sum_x[i])
                        self.observed_x_x[i] = np.sum(sum_x_x[i])
                ### if we are doing means and standards on a per-channel basis...
                if mode == "per":
                    if self.ori == "hwc":
                        big_n = self.dims[0] * self.dims[1] * self.full_data.shape[0]
                    else:
                        big_n = self.dims[1] * self.dims[2] * self.full_data.shape[0]
                    for i in range(self.nchannels):
                        self.observed_mean[i] = self.observed_x[i] / big_n
                        self.observed_std[i] = math.sqrt(
                            (self.observed_x_x[i] / big_n) - math.pow(observed_x[i] / big_n, 2))
                ### otherwise if we are doing means and standards over the entire image...
                elif mode == "global":
                    big_n = self.dims[0] * self.dims[1] * self.dims[2] * self.full_data.shape[0]
                    x_total = self.observed_x[0]
                    x_x_total = self.observed_x_x[0]
                    for i in range(1, self.nchannels):
                        x_total += self.observed_x[i]
                        x_x_total += self.observed_x_x[i]
                        #elf.observed_x[0] + self.observed_x[1] + self.observed_x[2]
                        #x_x_total = self.observed_x_x[0] + self.observed_x_x[1] + self.observed_x_x[2]
                    self.obsered_mean = [x_total / big_n] * self.nchannels
                    self.obsered_std = [math.sqrt((x_x_total / big_n) - math.pow(x_total / big_n, 2))] * self.nchannels
                print("computed observed means, std deviations")
                self.m_s_computed = True
            else:
                for i in range(self.full_data.shape[0]):
                    self.img_memory[i] = self.load_dcube(self.path_prefix + self.X_img_ref[i])
                print("loaded", self.full_data.shape[0], "images.")

    ### helper function to create fake means and stds arrays
    def blank_fake_ms(self, size):
        m = np.zeros(size)
        s = np.ones(size)
        return [m, s]

    def compute_mean_stds(self, mode="per"):
        if self.m_s_computed:
            print("warning... means and standards have already been computed")
        if mode == "per" or mode == "global":
            fake_m_s = self.blank_fake_ms(self.nchannels)
            #fake_m_s = [np.array([0, 0, 0]), np.array([1, 1, 1])]
            temp_img = np.zeros(self.dims)
            sum_x = np.zeros(self.dims)
            sum_x_x = np.zeros(self.dims)
            for i in range(self.full_data.shape[0]):
                temp_img = self.load_dcube(self.path_prefix + self.X_ref[i], fake_m_s, skip=True)
                #self.load_dcube(self.path_prefix + self.X_img_ref[i], fake_m_s, skip=True).astype('int64')
                sum_x += temp_img
                sum_x_x += np.multiply(temp_img, temp_img)
                #print("**")
                #print(temp_img)
                #print(np.multiply(temp_img, temp_img))
                #print("**")
            for i in range(self.nchannels):
                if self.ori == "hwc":
                    self.observed_x[i] = np.sum(sum_x[:,:,i])
                    self.observed_x_x[i] = np.sum(sum_x_x[:,:,i])
                else:
                    self.observed_x[i] = np.sum(sum_x[i])
                    self.observed_x_x[i] = np.sum(sum_x_x[i])
            if mode == "per":
                if self.ori == "hwc":
                    big_n = self.dims[0] * self.dims[1] * self.full_data.shape[0]
                else:
                    big_n = self.dims[1] * self.dims[2] * self.full_data.shape[0]
                for i in range(self.nchannels):
                    self.observed_mean[i] = self.observed_x[i]/big_n
                    #print("**")
                    #print(big_n)
                    #print(self.observed_mean[i])
                    #print(self.observed_x_x[i])
                    #print(self.observed_x[i])
                    #print("**")
                    self.observed_std[i] = math.sqrt((self.observed_x_x[i]/big_n) - (self.observed_x[i]/big_n)**2)
            elif mode == "global":
                big_n = self.dims[0] * self.dims[1] * self.dims[2]  * self.full_data.shape[0]
                x_total = self.overved_x[0]
                x_x_total = self.observed_x_x[0]
                for i in range(1, self.dims):
                    x_total += self.observed_x[i]
                    x_x_total += self.observed_x_x[i]
                    #x_total = self.observed_x[0] + self.observed_x[1] + self.observed_x[2]
                    #x_x_total = self.observed_x_x[0] + self.observed_x_x[1] + self.observed_x_x[2]
                self.obsered_mean = [x_total/big_n] * self.nchannels
                self.obsered_std = [math.sqrt((x_x_total/big_n) - math.pow(x_total/big_n, 2))] * self.nchannels
            self.m_s_computed = True
        else:
            print("invalid mean/std mode")
        ### done to here

    def get_or_compute_m_s(self, mode_in="per"):
        if self.m_s_computed:
            return [np.array(self.observed_mean), np.array(self.observed_std)]
        else:
            self.compute_mean_stds(mode=mode_in)
            return [np.array(self.observed_mean), np.array(self.observed_std)]

    def get_observed_m_s(self):
        return [np.array(self.observed_mean), np.array(self.observed_std)]

    def apply_observed_m_s(self):
        self.mean_stds = [np.array(self.observed_mean), np.array(self.observed_std)]
        if not self.mem_sensitive:
            for i in range(self.full_data.shape[0]):
                for j in range(self.nchannels):
                    if self.ori == "hwc":
                        self.img_memory[i,:,:,j] = (self.img_memory[i,:,:,j]-self.mean_stds[0][j] / self.mean_stds[1][j])
                    else:
                        self.img_memory[i, j] = (self.img_memory[i, j] - self.mean_stds[0][j]) / self.mean_stds[1][j]
        ### done to here

    def apply_given_m_s(self, m_s):
        self.mean_stds = m_s
        if not self.mem_sensitive:
            for i in range(self.full_data.shape[0]):
                for j in range(self.nchannels):
                    ### NOTE --- LEFT OFF HERE
                    if self.ori == "hwc":
                        self.img_memory[i,:,:,j] = (self.img_memory[i,:,:,j] - self.mean_stds[0][j])/self.mean_stds[1][j]
                    else:
                        self.img_memory[i, j] = (self.img_memory[i, j] - self.mean_stds[0][j]) / self.mean_stds[1][j]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_dcube(self, cube_loc, m_s="use_standard", skip=False):
        if m_s == "use_standard":
            m_s = self.mean_stds
        #image = np.array()
        ## need to assume these are square probably
        ### TODO - generalize this
        image_in = np.genfromtxt(cube_loc, delimiter=',')
        if self.ori == "hwc":
            image = image_in.reshape(int(math.sqrt(image_in.shape[0])),int(math.sqrt(image_in.shape[0])),
                    image_in.shape[1])
        else:
            image = image_in.reshape(image_in.shape[0], int(math.sqrt(image_in.shape[1])),
                    int(math.sqrt(image_in.shape[1])))
        #image = np.array(Image.open(img_loc))[:, :, :3]
        # print(image.shape, m_s, self.dataname)
        # if not isinstance(image, np.ndarray):
        #    if image == None:
        #        print("uhoh")
        if not skip:
            if self.ori == "hwc":
                for i in range(image.shape[2]):
                    if m_s[1][i] != 0:
                        image[:,:,i] = (image[:,:,i] - m_s[0][i]) / m_s[1][i]
            else:
                for i in range(image.shape[0]):
                    image[i] = (image[i] - m_s[0][i]) / m_s[1][i]
                    #image[:, :, i] = (image[:, :, i] - m_s[0][i]) / m_s[1][i]

        return image

    def get_n_samples (self):
        return len(self.y)

    #def predict_mode (self, pmode):


    def set_return (self, rmode):
        if rmode == "x" or rmode == "y" or rmode == "both":
            self.return_format = rmode
        else:
            print("cannot set return mode - invalid mode provided")
        return self

    def __len__ (self):
        return self.lenn

    def __getitem__(self, idx):
        # return picture data batch
        ret_indices = self.indexes[idx * self.batch_size: min(((idx + 1) * self.batch_size), self.full_data.shape[0])]
        if self.mem_sensitive:
            # print(ret_indices)
            ret_imgs = np.zeros((len(ret_indices), self.dims[0], self.dims[1], self.dims[2]))
            for i in range(len(ret_indices)):
                # print("img_ref=", self.path_prefix+self.X_img_ref[i])
                ret_imgs[i] = self.load_dcube(self.path_prefix + self.X_ref[ret_indices[i]])
            #if self.return_format == "both":
            #    return ret_imgs, self.y[ret_indices]
            if self.return_format == "x":
                return ret_imgs
            elif self.return_format == "y":
                return self.y[ret_indices]
            else:
                return ret_imgs, self.y[ret_indices]
        else:
            if self.return_format == "x":
                return self.img_memory[ret_indices]
            elif self.return_format == "y":
                return self.y[ret_indices]
            else:
                return self.img_memory[ret_indices], self.y[ret_indices]
