from osgeo import gdal
import numpy as np
import h5py
import os
import sys
import math
import raster_helpers
from scipy.signal import convolve2d
from multiprocessing import Pool

### BASIC DATA PROCESSING PREPARATION
###
###

### compute expected cube size
def compute_expected_cube_sizes(y_layer_max, cube_res):
    expected_cube_size = []
    for i in range(len(cube_res)):
        if cube_res[i] == y_layer_max:
            expected_cube_size.append(1)
        else:
            expected_cube_size.append(y_layer_max // cube_res[i] + 1)
    print("- computed expected cube sizes: ", expected_cube_size)
    return expected_cube_size

### make buffer around smaller data layers
def make_buffer(buffer_fill, layer_data, cube_res, y_base):
    buffer_dist = []
    for i in range(len(cube_res)):
        if i != y_base:
            localvals = np.array(layer_data[i])
            buffer_dist.append(cube_res[y_base] // (2*cube_res[i]) + 1)
            layer_data[i] = np.zeros((layer_data[i].shape[0] + (buffer_dist[i] * 2),
                                      layer_data[i].shape[1] + (buffer_dist[i] * 2)))
            layer_data[i] = layer_data[i] + buffer_fill
            layer_data[i][buffer_dist[i]:buffer_dist[i]+localvals.shape[0],
                          buffer_dist[i]:buffer_dist[i]+localvals.shape[1]] = localvals
        else:
            buffer_dist.append(0)
    print("- computed buffer distances: ", buffer_dist)
    return buffer_dist, layer_data

def reduce_from_total_dims(expected_cube_size, expected_sample_to, x_layers):
    total_params_cube = 0
    total_params_sample = 0
    for i in range(len(expected_sample_to)):
        if i in x_layers:
            total_params_cube += expected_cube_size[i] ** 2
            total_params_sample += expected_sample_to[i] ** 2
    ratio = math.sqrt(total_params_cube / total_params_sample)
    for i in range(len(expected_sample_to)):
        if i in x_layers:
            expected_sample_to[i] = int(expected_sample_to[i] * ratio)
    print("- reduced sampling dimension to match natural total parameters:", expected_sample_to[0])
    print("- new sample dimensions: ", expected_sample_to)
    return expected_sample_to


### compute offsets for sample generation
def compute_offsets(expected_cube_size, layer_data):
    center_offset = []
    half_offset = []
    for i in range(len(layer_data)):
        center_offset.append(expected_cube_size[i] // 2)
        if expected_cube_size[i] % 2 == 0:
            half_offset.append(0.5)
        else:
            half_offset.append(0)
    print("- computed sampling offsets")
    return center_offset, half_offset

### DEALING WITH REFERENCE SYSTEMS
###
###

### convert from coordinates to indices
def geo_idx(cx, cy, geopack): #ulh, ulv, psh, psv):
    ulh, ulv, psh, psv = geopack
    ix = (cx - ulh) / psh
    iy = (ulv - cy) / psv
    return ix, iy

### convert from indices to coordinates
def idx_geo(ix, iy, geopack): #ulh, ulv, psh, psv):
    ulh, ulv, psh, psv = geopack
    cx = ulh + (ix * psh)
    cy = ulv - (iy * psv)
    return cx, cy

### ROUNDUP 1: FIND LEGAL SAMPLES
###
###

### determine whether a sample is legal
### legality requires no missing data in the sample area
def roundup_layer_1(k, base_idx, buffer_ignore, crs_list, yloc, nd_vals, expected_cube_size, layer_data,
                    buffer_dist, half_offset, center_offset, buffer_fill):
    bi, bj = base_idx
    geo_ctr = idx_geo(bi + 0.5, bj + 0.5, crs_list[yloc])
    if k == yloc:
        if layer_data[k][bi, bj] != nd_vals[k] and not np.isnan(layer_data[k][bi, bj]).any():
            return True
    elif expected_cube_size[k] == 1:
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        #print(tidi, tidj)
        #if layer_data[k][int(tidi)+buffer_dist[k], int(tidj)+buffer_dist[k]] != nd_vals[k] and \
        if layer_data[k][int(tidi), int(tidj)] != nd_vals[k] and \
                not np.isnan(layer_data[k][int(tidi), int(tidj)]).any():
                #not np.isnan(layer_data[k][int(tidi)+buffer_dist[k], int(tidj)+buffer_dist[k]]).any():
            return True
    else:
        ### need to determine UL
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])

        sulx = int(tidi + half_offset[k]) - center_offset[k]
        suly = int(tidj + half_offset[k]) - center_offset[k]

        temp = layer_data[k][sulx:sulx+expected_cube_size[k],
                             suly:suly+expected_cube_size[k]].reshape(-1)
        if nd_vals[k] not in temp and not np.isnan(temp).any():
            temp = temp[temp != buffer_ignore]
            if len(temp) > 0:
                return True
    return False

### TODO - parallelize
def compile_legal_samples(expected_cube_size, layer_data, y_base, cube_res, buffer_fill,
                          layer_crs, layer_nodata, buffer_dist, half_offset, center_offset):
    ### gather legal samples
    ### ...batch based on regions?
    legal_sample_idx_list = []
    guide_shape = layer_data[y_base].shape
    print("- compile samples progress ", end="", flush=True)
    for i in range(guide_shape[0]):
        if i % (guide_shape[0] // 10) == 0:
            print("-", end="", flush=True)
        if i == guide_shape[0] - 1:
            print("->| done")
        for j in range(guide_shape[1]):
            ### determine if nodata value is involved, and ignore buffer fill values
            all_ok = True
            tmins = []
            tmaxs = []
            for k in range(len(cube_res)):
                ### check individual layer for nodata
                layer_k_np = roundup_layer_1(k, (i, j), buffer_fill, layer_crs, y_base,
                                             layer_nodata, expected_cube_size, layer_data,
                                             buffer_dist, half_offset, center_offset,
                                             buffer_fill)
                if layer_k_np == False:
                    all_ok = False
                    break
            if all_ok:
                legal_sample_idx_list.append((i, j))
    print("- rounded up layers: ", len(legal_sample_idx_list))
    print("  - maximum legal samples = ", guide_shape[0] * guide_shape[1])
    return legal_sample_idx_list, guide_shape

### save list of legal samples for future use
def save_legal_sample_ids(legal_sample_idx_list, fold_name):
    legal_idx_save = np.array(legal_sample_idx_list)
    np.savetxt(fold_name + "/legal_ids.csv", legal_idx_save, delimiter=",")
    print("- saved sample coords")

### load list of legal samples
def load_legal_sample_ids(fold_name):
    return np.genfromtxt(fold_name + "/legal_ids.csv", delimiter=",")
### TRAIN/VAL/TEST SPLIT METHODS
###
###

def create_block_buffer_mask(block_mask, buffer):
    ### 0 for illegal, 1 for legal, 2 for taken
    ### set 1-values to 0 and non-1-values to 1, then run a kernel over
    block_buffer_mask = np.array(block_mask)
    block_buffer_mask[block_mask != 2] = 0
    block_buffer_mask[block_mask == 2] = 1
    #kernelr = np.ones(((buffer * 2) + 1, 1))
    #kernelc = np.ones((1, (buffer * 2) + 1))
    kernelrc = np.ones(((buffer * 2) + 1, (buffer * 2) + 1))
    #block_buffer_mask = convolve2d(block_buffer_mask, kernelr, mode='same')
    #block_buffer_mask = convolve2d(block_buffer_mask, kernelc, mode='same')
    bbm_1 = np.int64(convolve2d(block_buffer_mask, kernelrc, mode='same') > 0)
    #conved = np.int64(block_buffer_mask > 0)
    #conved[block_mask == 0] = 1
    bbm_1[block_mask == 0] = 1
    sample_from = np.where(bbm_1 == 0)
    if len(sample_from[0]) == 0:
        ### no samples to go from! uh oh!
        return False
    return sample_from, bbm_1



### TODO - CHECK RETURNS
### mask legal samples for test set
def block_split_test_iter(split_blocks_nregions, guide_shape, blocksizes, block_mask_in, split_blocks_buffer,
                          each_err_test, approx_test_each):
    block_mask = block_mask_in.copy()
    ### place blocks over region
    ### update - make better block mask
    nplaced = 0
    carrier = 0
    print("- test creation progress ", end="")
    while nplaced < split_blocks_nregions:
        ### narrow list of good options to randomly sample from
        narrow_list, bbm = create_block_buffer_mask(block_mask, split_blocks_buffer)
        if narrow_list == False:
            print("- oops! no legal locations to begin drawing boxes from. Restarting...")
            return block_split_test_iter(split_blocks_nregions, guide_shape, blocksizes, block_mask,
                                         split_blocks_buffer, each_err_test, approx_test_each)
        idrand = int(np.random.uniform(0, len(narrow_list[0])))
        ruli = narrow_list[0][idrand]
        rulj = narrow_list[1][idrand]
        if block_mask[ruli, rulj] == 1:
            nplaced += 1
            if nplaced % (split_blocks_nregions // 10) == 0:
                print("-", end="", flush=True)
            carrier += each_err_test
            reqd = int(approx_test_each) + int(carrier)
            carrier = carrier % 1
            ### good to go! start with centerpoint
            total1 = 0
            tradius = -1
            iteri = (int(np.random.uniform(0, 2)) * 2) - 1
            iterj = (int(np.random.uniform(0, 2)) * 2) - 1
            while total1 < reqd:
                tradius += 2
                ### need to account for oob
                for i in range(tradius):
                    for j in range(tradius):
                        ti = ruli - (iteri * (tradius // 2)) + (iteri * i)
                        tj = rulj - (iterj * (tradius // 2)) + (iterj * j)
                        if ti >= 0 and ti < block_mask.shape[0] and tj >= 0 and tj < block_mask.shape[1]:
                            if block_mask[ti, tj] == 1:
                                total1 += 1
                                block_mask[ti, tj] = 2
                                if total1 >= reqd:
                                    break
                    if total1 >= reqd:
                        break
        else:
            print("x", block_mask[ruli, rulj], bbm[ruli, rulj], end="")
            #pass
    print(">| done")
    return block_mask

### mask legal samples for train/val sets
def block_split_tranval_iter(block_mask, split_blocks_nregions, split_outer_buffer, guide_shape,
                             split_blocks_buffer, blocksizes, each_err_val, approx_val_each):
    split_mask_i = block_mask.copy()
    ### place blocks over region
    nplaced_i = 0
    carrier = 0
    while nplaced_i < split_blocks_nregions:
        narrow_list, bbm = create_block_buffer_mask(split_mask_i, split_blocks_buffer)
        if narrow_list == False:
            print("- oops! no legal locations to begin drawing boxes from. Restarting...")
            return block_split_test_iter(split_blocks_nregions, guide_shape, blocksizes, block_mask,
                                         split_blocks_buffer, each_err_val, approx_val_each)
        idrand = int(np.random.uniform(0, len(narrow_list[0])))
        ruli = narrow_list[0][idrand]
        rulj = narrow_list[1][idrand]
        if split_mask_i[ruli, rulj] == 1:
            nplaced_i += 1
            if nplaced_i % (split_blocks_nregions // 10) == 0:
                print("-", end="", flush=True)
            carrier += each_err_val
            reqd = int(approx_val_each) + int(carrier)
            carrier = carrier % 1
            ### good to go! start with centerpoint
            total1 = 0
            tradius = -1
            iteri = (int(np.random.uniform(0, 2)) * 2) - 1
            iterj = (int(np.random.uniform(0, 2)) * 2) - 1
            while total1 < reqd:
                tradius += 2
                ### need to account for oob
                for i in range(tradius):
                    for j in range(tradius):
                        ti = ruli - (iteri * (tradius // 2)) + (iteri * i)
                        tj = rulj - (iterj * (tradius // 2)) + (iterj * j)
                        if ti >= 0 and ti < split_mask_i.shape[0] and tj >= 0 and tj < split_mask_i.shape[1]:
                            if split_mask_i[ti, tj] == 1:
                                total1 += 1
                                split_mask_i[ti, tj] = 2
                                if total1 >= reqd:
                                    break
                    if total1 >= reqd:
                        break
        else:
            print("x", split_mask_i[ruli, rulj], bbm[ruli, rulj], end="")
            #pass
    print(">| done")
    return split_mask_i

### Split in geographic blocks
def block_split(legal_sample_idx_list, partition, split_blocks_nregions, guide_shape,
                split_blocks_buffer, fold_name, layer_proj, y_base, n_splits,
                split_outer_buffer):
    val_fold_indices = []
    train_fold_indices = []
    n_test_samples = int(len(legal_sample_idx_list) * partition[0])
    approx_each = n_test_samples / split_blocks_nregions
    temp_each_sqrt = math.ceil(math.sqrt(approx_each))
    ### calculate block sizes from parameters
    minerr = 100000
    minoff = 0

    for i in range(max(temp_each_sqrt // 5, 1)):
        err1 = abs(
            (math.ceil(approx_each / (temp_each_sqrt - (i - (temp_each_sqrt // 10)))) * temp_each_sqrt) - approx_each)
        if err1 < minerr:
            minerr = err1
            minoff = i - (temp_each_sqrt // 10)
    blocksizes = (temp_each_sqrt - minoff, math.ceil(approx_each / (temp_each_sqrt - minoff)))
    underest = (blocksizes[0] * blocksizes[1]) - (n_test_samples / split_blocks_nregions)
    intoffset = underest - int(underest)
    n_val_samples = int(len(legal_sample_idx_list) * partition[1])
    approx_val_each = n_val_samples / split_blocks_nregions
    each_err_val = approx_val_each - int(approx_val_each)

    n_test_samples = int(len(legal_sample_idx_list) * partition[0])
    approx_test_each = n_test_samples / split_blocks_nregions
    each_err_test = approx_test_each - int(approx_test_each)

    print("- computed test set block sizes")
    print("  - total_blocks:", n_test_samples, "approx. each:", approx_each, "sqrt:", temp_each_sqrt)
    print("  - blocksizes:", blocksizes, "block each:", blocksizes[0]*blocksizes[1], "est:", underest)
    print("  - guid_shape", guide_shape)

    ### make mask grid
    block_mask = np.zeros(guide_shape)
    metaindex_arr = np.zeros(guide_shape)
    for i in range(len(legal_sample_idx_list)):
        ti, tj = legal_sample_idx_list[i]
        block_mask[ti, tj] = 1
        metaindex_arr[ti, tj] = i

    block_mask = block_split_test_iter(split_blocks_nregions, guide_shape, blocksizes, block_mask, split_blocks_buffer,
                          each_err_test, approx_test_each)

    overlap3 = np.where(block_mask == 2)
    test_indices = metaindex_arr[overlap3]
    remaining_indices = metaindex_arr[np.where(block_mask == 1)]
    print("- found test indices:", len(test_indices), "out of", n_test_samples, "desired")

    ### make tifs for visual confirmation
    os.system("rm " + fold_name + "/fold_box_geotifs/*")
    os.system("mkdir " + fold_name + "/fold_box_geotifs")
    raster_helpers.save_raster(fold_name + "/fold_box_geotifs", "test_extent", block_mask, layer_proj[y_base][0],
                               layer_proj[y_base][1], -1)

    print("- saved test geotif visualizations")

    block_mask[overlap3] = 0

    ### now do xval-splits
    for ii in range(n_splits):
        print("- fold", ii, "creation progress ", end="")
        split_mask_i = block_split_tranval_iter(block_mask, split_blocks_nregions, split_outer_buffer,
                                                guide_shape, split_blocks_buffer, blocksizes, each_err_val,
                                                approx_val_each)
        overlap4 = np.where(split_mask_i == 2)
        val_fold_indices.append(metaindex_arr[overlap4])
        train_fold_indices.append(metaindex_arr[np.where(split_mask_i == 1)])
        np.savetxt(fold_name + "/train_" + str(ii) + ".csv", train_fold_indices[-1], delimiter=",")
        np.savetxt(fold_name + "/val_" + str(ii) + ".csv", val_fold_indices[-1], delimiter=",")

        ### make geotif for visual inspection
        raster_helpers.save_raster(fold_name + "/fold_box_geotifs", "val_extent_" + str(ii), split_mask_i,
                                   layer_proj[y_base][0], layer_proj[y_base][1], -1)
    print("- found training and validation indices")
    print("- saved training and validation indices")
    print("- saved training and validation geotif visualizations")
    np.savetxt(fold_name + "/test.csv", test_indices, delimiter=",")
    np.savetxt(fold_name + "/remaining.csv", remaining_indices, delimiter=",")
    print("- saved test indices")

    return test_indices, remaining_indices, train_fold_indices, val_fold_indices

def load_splits(fold_name, n_splits):
    test_indices = None
    remaining_indices = None
    train_fold_indices = []
    val_fold_indices = []
    test_indices = np.genfromtxt(fold_name + "/test.csv", delimiter=",")
    remaining_indices = np.genfromtxt(fold_name + "/remaining.csv", delimiter=",")
    for i in range(n_splits):
        train_fold_indices.append(np.genfromtxt(fold_name + "/train_" + str(i) + ".csv", delimiter=","))
        val_fold_indices.append(np.genfromtxt(fold_name + "/val_" + str(i) + ".csv", delimiter=","))
    return test_indices, remaining_indices, train_fold_indices, val_fold_indices

def fullrand_split(legal_sample_idx_list, partition, n_splits, fold_name, meta_indices):
    train_fold_indices = []
    val_fold_indices = []
    np.random.shuffle(meta_indices)
    test_indices = meta_indices[:int(len(meta_indices) * partition[0])]
    remaining_indices = meta_indices[int(len(meta_indices) * partition[0]):]
    np.random.shuffle(remaining_indices)
    print("- found test indices")
    for i in range(n_splits):
        np.random.shuffle(remaining_indices)
        val_fold_indices.append(remaining_indices[:int(len(remaining_indices) * partition[1])])
        train_fold_indices.append(remaining_indices[int(len(remaining_indices) * partition[1]):])
        np.savetxt(fold_name + "/train_" + str(i) + ".csv", train_fold_indices[-1], delimiter=",")
        np.savetxt(fold_name + "/val_" + str(i) + ".csv", val_fold_indices[-1], delimiter=",")
    print("- found training and validation indices")
    print("- saved training and validation indices")
    np.savetxt(fold_name + "/test.csv", test_indices, delimiter=",")
    np.savetxt(fold_name + "/remaining.csv", remaining_indices, delimiter=",")
    print("- saved test indices")

### should always return a 2d np array...
def roundup_layer_2(k, base_idx, crs_list, yloc, layer_data, expected_cube_size, buffer_dist, half_offset,
                    center_offset, sample_to_res):
    bi, bj = base_idx
    geo_ctr = idx_geo(bi + 0.5, bj + 0.5, crs_list[yloc])
    if k == yloc:
        return layer_data[[[bi]], [[bj]]]
    elif expected_cube_size == 1:
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        return layer_data[[[int(tidi)+buffer_dist]],[[int(tidj)+buffer_dist]]]
    else:
        ### need to determine UL
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        tidi += buffer_dist
        tidj += buffer_dist
        sulx = int(tidi + half_offset) - center_offset
        suly = int(tidj + half_offset) - center_offset
        return layer_data[sulx:sulx+expected_cube_size,
                             suly:suly+expected_cube_size]

def roundup_layer_3(k, base_idx, crs_list, yloc, layer_data, expected_cube_size, buffer_dist, half_offset,
                    center_offset, sample_to_res):
    bi, bj = base_idx
    geo_ctr = idx_geo(bi + 0.5, bj + 0.5, crs_list[yloc])
    if k == yloc:
        return layer_data[[[bi]], [[bj]]]
    elif sample_to_res == 1:
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        return layer_data[[[int(tidi)+buffer_dist]], [[int(tidj)+buffer_dist]]]
    else:
        ### need to determine UL
        ### have center saved...?
        ### sample n points...
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        tidii = tidi + buffer_dist
        tidjj = tidj + buffer_dist
        ### UL is obtained...
        sulx = int(tidi + half_offset) - center_offset
        suly = int(tidj + half_offset) - center_offset
        stepi = ((sulx-tidi)*2)/(sample_to_res - 1)
        stepj = ((suly - tidj) * 2) / (sample_to_res - 1)
        sampled = np.zeros((sample_to_res, sample_to_res))
        for i in range(sample_to_res):
            for j in range(sample_to_res):
                sampled[i, j] = layer_data[int(i * stepi), int(j * stepj)]
        return sampled

def make_pyramid_layer(pyparams):
    fold_name, h5_chunk_size, expected_cube_size, legal_sample_idx_list, k, layer_crs, test_indices, buffer_fill,\
    n_splits, train_fold_indices, sample_min, sample_max, fold_min, fold_max, y_base, sample_to_res, layer_data,\
    buffer_dist, half_offset, center_offset = pyparams
    ### make pyramid layer

    os.system("rm " + fold_name + "/layer_" + str(k) + ".h5")
    h5_data_file_i = h5py.File(fold_name + "/layer_" + str(k) + ".h5", "a")
    h5_file_set_i = h5_data_file_i.create_dataset("data", (h5_chunk_size, sample_to_res, sample_to_res),
                                        maxshape=(None, sample_to_res, sample_to_res),
                                        chunks=(h5_chunk_size, sample_to_res, sample_to_res))

    h5_chunk_i = np.zeros((h5_chunk_size, sample_to_res, sample_to_res))
    h5_chunk_i.fill(-1)

    h5_chunk_counter = 0

    if sample_to_res == expected_cube_size:
        roundup_active = roundup_layer_2a
        print("-", k, "running roundup 2a")
    else:
        roundup_active = roundup_layer_3a
        print("- running roundup 3a")

    ### deal with sampling dims...?


    for i in range(len(legal_sample_idx_list)):
        result = roundup_active(k, legal_sample_idx_list[i], layer_crs, y_base, layer_data, expected_cube_size,
                                 buffer_dist, half_offset, center_offset, sample_to_res)
        ### add to h5
        h5_chunk_i[h5_chunk_counter % h5_chunk_size, :, :] = result
        if i not in test_indices:
            ### get scaling info
            mmcheck = result[result != buffer_fill]
            if len(mmcheck > 0):
                nanmin = np.nanmin(mmcheck)
                nanmax = np.nanmax(mmcheck)
                ### adjust min/max for combined data
                sample_min = min(nanmin, sample_min)
                sample_max = max(nanmax, sample_max)
                ### adjust min/max for individual splits
                for j in range(n_splits):
                    if i in train_fold_indices[j]:
                        fold_min[j] = min(nanmin, fold_min[j])
                        fold_max[j] = max(nanmax, fold_max[j])

        h5_chunk_counter += 1
        if h5_chunk_counter % h5_chunk_size == 0:
            ### resize to current number of items
            h5_file_set_i.resize(h5_chunk_counter, axis=0)
            ### set values from current chunk
            h5_file_set_i[h5_chunk_counter - h5_chunk_size:h5_chunk_counter, :, :] = \
                np.array(h5_chunk_i[:, :, :])
            h5_chunk_i = np.zeros(h5_chunk_i.shape)
            h5_chunk_i.fill(-1)

    ### finish remainder of h5 chunk and save
    h5_file_set_i.resize(h5_chunk_counter, axis=0)
    h5_file_set_i[h5_chunk_counter - (h5_chunk_counter % h5_chunk_size):h5_chunk_counter, :, :] = \
        np.array(h5_chunk_i[:h5_chunk_counter % h5_chunk_size, :, :])
    h5_data_file_i.close()

    return sample_min, sample_max, fold_min, fold_max


def make_pyramids_main(cube_res, fold_name, h5_chunk_size, expected_cube_size,
                       legal_sample_idx_list, layer_crs, y_base, test_indices, buffer_fill, n_splits,
                       layer_data, train_fold_indices, sample_to_res, buffer_dist, half_offset, center_offset):

    print("- set up h5 datasets")

    ### for folds, organized as [fold][layer]
    samples_mins = []
    samples_maxs = []
    folds_mins = []
    folds_maxs = []
    for i in range(len(layer_data)):
        folds_mins.append([])
        folds_maxs.append([])
        for j in range(n_splits):
            folds_mins[i].append(float("inf"))
            folds_maxs[i].append(float("-inf"))

    for i in range(len(layer_data)):
        samples_mins.append(float("inf"))
        samples_maxs.append(float("-inf"))

    pyramid_params = []
    for i in range(len(cube_res)):
        pyramid_params.append([fold_name, h5_chunk_size, expected_cube_size[i], legal_sample_idx_list, i,
                               layer_crs, test_indices, buffer_fill, n_splits, train_fold_indices,
                               samples_mins[i], samples_maxs[i], folds_mins[i], folds_maxs[i], y_base,
                               sample_to_res[i], layer_data[i], buffer_dist[i], half_offset[i], center_offset[i]])
    parallel = True
    if parallel:
        with Pool(None) as mpool:
            resarr = mpool.map(make_pyramid_layer, pyramid_params)
    else:
        pass

    print("- reformatted data to h5")
    print("- saved h5 files")
    print("- calculated min/max normalization parameters")
    for i in range(len(resarr)):
        sample_min, sample_max, fold_min, fold_max = resarr[i]
        folds_mins[i] = fold_min
        folds_maxs[i] = fold_max
        samples_mins[i] = sample_min
        samples_maxs[i] = sample_max

    combined_mins = np.array(samples_mins)
    combined_maxs = np.array(samples_maxs)
    np.savetxt(fold_name + "/norm_layer_mins_combined.csv", combined_mins, delimiter=",")
    np.savetxt(fold_name + "/norm_layer_maxs_combined.csv", combined_maxs, delimiter=",")

    fold_np_mins = np.array(folds_mins).transpose()
    fold_np_maxs = np.array(folds_maxs).transpose()
    for j in range(n_splits):
        np.savetxt(fold_name + "/norm_layer_mins_fold_" + str(j) + ".csv", fold_np_mins[j], delimiter=",")
        np.savetxt(fold_name + "/norm_layer_maxs_fold_" + str(j) + ".csv", fold_np_maxs[j], delimiter=",")
    print("- saved min/max normalization parameters")
    print("- done")

### should always return a 2d np array...
def roundup_layer_2a(k, base_idx, crs_list, yloc, layer_data, expected_cube_size, buffer_dist, half_offset,
                    center_offset, sample_to_res):
    bi, bj = base_idx
    geo_ctr = idx_geo(bi + 0.5, bj + 0.5, crs_list[yloc])
    if k == yloc:
        return layer_data[[[bi]], [[bj]]]
    elif expected_cube_size == 1:
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        return layer_data[[[int(tidi)]],[[int(tidj)]]]
    else:
        ### need to determine UL
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        sulx = int(tidi + half_offset) - center_offset
        suly = int(tidj + half_offset) - center_offset
        return layer_data[sulx:sulx+expected_cube_size, suly:suly+expected_cube_size]

### should always return a 2d np array...
def roundup_layer_3a(k, base_idx, crs_list, yloc, layer_data, expected_cube_size, buffer_dist, half_offset,
                    center_offset, sample_to_res):
    bi, bj = base_idx
    geo_ctr = idx_geo(bi + 0.5, bj + 0.5, crs_list[yloc])
    if k == yloc:
        return layer_data[[[bi]], [[bj]]]
    elif expected_cube_size == 1:
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        #return layer_data[[[int(tidi)-buffer_dist]],[[int(tidj)-buffer_dist]]]
        return layer_data[[[int(tidi)]], [[int(tidj)]]]
    else:
        ### need to determine UL
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        result = np.zeros((sample_to_res, sample_to_res))
        sulx = int(tidi + half_offset - 0.5) - center_offset
        suly = int(tidj + half_offset - 0.5) - center_offset
        iterstep = (2 * (center_offset - half_offset) + 1) / (sample_to_res - 1)
        for i in range(sample_to_res):
            for j in range(sample_to_res):
                result[i, j] = layer_data[int(sulx + i*iterstep), int(suly + j*iterstep)]
        return result

def pyramid_nonparallel(fold_name, h5_chunk_size, expected_cube_size, n_splits, layer_data, meta_indices,
                        cube_res, legal_sample_idx_list, layer_crs, y_base, test_indices,
                        train_fold_indices, buffer_fill, sample_to_res, buffer_dist, half_offset,
                        center_offset):
    ### setup for h5
    h5_data_files = []
    h5_file_sets = []
    h5_chunks = []
    h5_chunk_counter = 0
    h5_chunkid = 0
    for i in range(len(cube_res)):
        os.system("rm " + fold_name + "/layer_" + str(i) + ".h5")
        h5_data_files.append(h5py.File(fold_name + "/layer_" + str(i) + ".h5", "a"))
        h5_file_sets.append(h5_data_files[i].create_dataset("data",
                                                            (h5_chunk_size, sample_to_res[i],
                                                             sample_to_res[i]),
                                                            maxshape=(
                                                            None, sample_to_res[i], sample_to_res[i]),
                                                            chunks=(h5_chunk_size, sample_to_res[i],
                                                                    sample_to_res[i])))
        h5_chunks.append(np.zeros((h5_chunk_size, sample_to_res[i], sample_to_res[i])))
        h5_chunks[-1].fill(-1)

    print("- set up h5 datasets")

    ### for folds, organized as [fold][layer]
    samples_mins = []
    samples_maxs = []
    folds_mins = []
    folds_maxs = []
    for i in range(len(layer_data)):
        folds_mins.append([])
        folds_maxs.append([])
        for j in range(n_splits):
            folds_mins[i].append(float("inf"))
            folds_maxs[i].append(float("-inf"))

    for i in range(len(layer_data)):
        samples_mins.append(float("inf"))
        samples_maxs.append(float("-inf"))

    if sample_to_res == expected_cube_size:
        roundup_active = roundup_layer_2a
        print("- running roundup 2a")
    else:
        roundup_active = roundup_layer_3
        print("- running roundup 3a")

    ### go back through combined list and compute min/max
    ### convert to h5 and save
    for i in range(len(meta_indices)):
        ### get data for h5
        ### if not in test, get min/max for normalization
        for k in range(len(cube_res)):
            result = roundup_active(k, legal_sample_idx_list[i], layer_crs, y_base, layer_data[k], expected_cube_size[k],
                                 buffer_dist[k], half_offset[k], center_offset[k], sample_to_res[k])
            ### add to h5
            h5_chunks[k][h5_chunk_counter % h5_chunk_size, :, :] = result
            if i not in test_indices:
                ### get scaling info
                ### a sample at layer i
                mmcheck = result[result != buffer_fill]
                if len(mmcheck > 0):
                    nanmin = np.nanmin(mmcheck)
                    nanmax = np.nanmax(mmcheck)
                    ### adjust min/max for combined data ## good
                    samples_mins[k] = min(nanmin, samples_mins[k])
                    samples_maxs[k] = max(nanmax, samples_maxs[k])
                    ### adjust min/max for individual splits
                    for j in range(n_splits):
                        if i in train_fold_indices[j]:
                            folds_mins[k][j] = min(nanmin, folds_mins[k][j])
                            folds_maxs[k][j] = max(nanmax, folds_maxs[k][j])

        h5_chunk_counter += 1
        if h5_chunk_counter % h5_chunk_size == 0:
            for k in range(len(cube_res)):
                ### resize to current number of items
                h5_file_sets[k].resize(h5_chunk_counter, axis=0)
                ### set values from current chunk
                h5_file_sets[k][h5_chunk_counter - h5_chunk_size:h5_chunk_counter, :, :] = \
                    np.array(h5_chunks[k][:, :, :])
                h5_chunks[k] = np.zeros(h5_chunks[k].shape)
                h5_chunks[k].fill(-1)

    print("- reformatted data to h5")

    ### finish remainder of h5 chunk and save
    for k in range(len(cube_res)):
        h5_file_sets[k].resize(h5_chunk_counter, axis=0)
        h5_file_sets[k][h5_chunk_counter - (h5_chunk_counter % h5_chunk_size):h5_chunk_counter, :, :] = \
            np.array(h5_chunks[k][:h5_chunk_counter % h5_chunk_size, :, :])
        h5_data_files[k].close()

    print("- saved h5 files")
    print("- calculated min/max normalization parameters")
    combined_mins = np.array(samples_mins)
    combined_maxs = np.array(samples_maxs)
    np.savetxt(fold_name + "/norm_layer_mins_combined.csv", combined_mins, delimiter=",")
    np.savetxt(fold_name + "/norm_layer_maxs_combined.csv", combined_maxs, delimiter=",")

    fold_np_mins = np.array(folds_mins).transpose()
    fold_np_maxs = np.array(folds_maxs).transpose()
    for j in range(n_splits):
        np.savetxt(fold_name + "/norm_layer_mins_fold_" + str(j) + ".csv", fold_np_mins[j], delimiter=",")
        np.savetxt(fold_name + "/norm_layer_maxs_fold_" + str(j) + ".csv", fold_np_maxs[j], delimiter=",")

    print("- saved min/max normalization parameters")
    print("- done")
    print("- saved min/max normalization parameters")
    print("- done")

def save_info_file(data_info, expected_sample_to, fold_name, n_splits, buffer_fill, data_input_crs,
                   np_random_seed):
    ### SAVE INFO
    with open(fold_name + "/info.txt", 'w') as infofile:
        infofile.write(str(n_splits) + "," + str(buffer_fill) + "," + data_input_crs +
                       "," + str(np_random_seed))
        for i in range(len(data_info)):
            infofile.write("\n" + str(expected_sample_to[i]) + "," + str(data_info[i][3]) + "," +
                           str(data_info[i][4]) + "," + str(data_info[i][5]))
    print("- saved info file")

def set_checkpoint(fold_name, checkpoint_number):
    with open(fold_name + "/checkpoint.txt", 'w') as checkpoint:
        checkpoint.write(str(checkpoint_number))
    print("- checkpoint:", checkpoint_number)

def get_checkpoint(fold_name):
    with open(fold_name + "/checkpoint.txt", 'r') as checkpoint:
        checkpoint_number = checkpoint.read()
    return int(checkpoint_number)