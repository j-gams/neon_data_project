### create_split.py

from osgeo import gdal
import numpy as np
import h5py
import os
import sys
import math
import raster_helpers

print("- loaded packages")

### EPSG 4326 WSG 84

### user parameters
### location, resolution, x/y, idx
data_info = [["../data/raster/srtm_clipped_co.tif",                 30,     "x", 0],
             ["../data/raster/nlcd2001_clipped_co_reproj.tif",      30,     "x", 1],
             ["../data/raster/nlcd2004_clipped_co_reproj.tif",      30,     "x", 2],
             ["../data/raster/nlcd2006_clipped_co_reproj.tif",      30,     "x", 3],
             ["../data/raster/nlcd2008_clipped_co_reproj.tif",      30,     "x", 4],
             ["../data/raster/nlcd2011_clipped_co_reproj.tif",      30,     "x", 5],
             ["../data/raster/nlcd2013_clipped_co_reproj.tif",      30,     "x", 6],
             ["../data/raster/nlcd2016_clipped_co_reproj.tif",      30,     "x", 7],
             ["../data/raster/nlcd_clipped_co_reproj.tif",          30,     "x", 8],
             ["../data/raster/nlcd2021_clipped_co_reproj.tif",      30,     "x", 9],
             ["../data/raster/aspect_clipped_co.tif",               30,     "x", 10],
             ["../data/raster/slope_clipped_co.tif",                30,     "x", 11],
             ["../data/raster/treeage_clipped_co_reproj.tif",       1000,   "x", 12],
             ["../data/raster/prism_precip_30y_800m_reproj.tif",    800,    "x", 13],
             ["../data/raster/prism_tempmin_30y_800m_reproj.tif",   800,    "x", 14],
             ["../data/raster/prism_tempmean_30y_800m_reproj.tif",  800,    "x", 15],
             ["../data/raster/prism_tempmax_30y_800m_reproj.tif",   800,    "x", 16],
             ["../data/raster/prism_vapormin_30y_800m_reproj.tif",  800,    "x", 17],
             ["../data/raster/prism_vapormax_30y_800m_reproj.tif",  800,    "x", 18],
             ["../data/raster/ecostresswue_clipped_co.tif",         70,     "y", 19],
             ["../data/raster/ecostressesi_clipped_co.tif",         70,     "y", 20],
             ["../data/raster/gedi_agforestbiomass_clipped_co.tif", 1000,   "y", 21]]
"""

data_info = [["../data/raster/srtm_clipped_co.tif",                 30,     "x", 0],
             ["../data/raster/nlcd2001_clipped_co_reproj.tif",      30,     "x", 1],
             ["../data/raster/nlcd2004_clipped_co_reproj.tif",      30,     "x", 2],
             ["../data/raster/nlcd2006_clipped_co_reproj.tif",      30,     "x", 3],
             ["../data/raster/nlcd2008_clipped_co_reproj.tif",      30,     "x", 4],
             ["../data/raster/nlcd2011_clipped_co_reproj.tif",      30,     "x", 5],
             ["../data/raster/nlcd2013_clipped_co_reproj.tif",      30,     "x", 6],
             ["../data/raster/nlcd2016_clipped_co_reproj.tif",      30,     "x", 7],
             ["../data/raster/nlcd_clipped_co_reproj.tif",          30,     "x", 8],
             ["../data/raster/nlcd2021_clipped_co_reproj.tif",      30,     "x", 9],
             ["../data/raster/aspect_clipped_co.tif",               30,     "x", 10],
             ["../data/raster/slope_clipped_co.tif",                30,     "x", 11],
             ["../data/raster/treeage_clipped_co_reproj.tif",       30,     "x", 12],
             ["../data/raster/prism_precip_30y_800m_reproj.tif",    30,     "x", 13],
             ["../data/raster/prism_tempmin_30y_800m_reproj.tif",   30,     "x", 14],
             ["../data/raster/prism_tempmean_30y_800m_reproj.tif",  30,     "x", 15],
             ["../data/raster/prism_tempmax_30y_800m_reproj.tif",   30,     "x", 16],
             ["../data/raster/prism_vapormin_30y_800m_reproj.tif",  30,     "x", 17],
             ["../data/raster/prism_vapormax_30y_800m_reproj.tif",  30,     "x", 18],
             ["../data/raster/ecostresswue_clipped_co.tif",         70,     "y", 19],
             ["../data/raster/ecostressesi_clipped_co.tif",         70,     "y", 20],
             ["../data/raster/gedi_agforestbiomass_clipped_co.tif", 1000,   "y", 21]]

"""
### split method and parameters
y_base = len(data_info) - 1
# how to divide geographic areas for train/val/test split
split_method = "blocks" # in {blocks, fullrand}
split_blocks_buffer = 10
split_blocks_nregions = 100
split_outer_buffer = 2

### partition parameters
n_splits = 1
partition = (0.3, 0.2)

### input data parameters
exclude = []
np_random_seed = 100807
fold_name = "../data/box_cube"
h5_chunk_size = 1000



### auto params
layer_locs = []
cube_res = []
y_layers = []
x_layers = []
layer_names = []
layer_nodata = []
layer_size = []
layer_crs = []
layer_proj = []
layer_data = []
for i in range(len(data_info)):
    if i not in exclude:
        layer_locs.append(data_info[i][0])
        cube_res.append(data_info[i][1])
        if data_info[i][2] == "x":
            x_layers.append(i)
        else:
            y_layers.append(i)

### import raster layers, get data and crs info
for item in layer_locs:
    layer_names.append(item.split("/")[-1].split(".")[0])
    layer_raster = gdal.Open(item)
    rasterband = layer_raster.GetRasterBand(1)
    layer_nodata.append(rasterband.GetNoDataValue())
    layer_size.append((layer_raster.RasterXSize, layer_raster.RasterYSize))
    tulh, tpxh, _, tulv, _, tpxv = layer_raster.GetGeoTransform()
    tpxv = abs(tpxv)
    layer_crs.append((tulh, tulv, tpxh, tpxv))
    layer_proj.append([layer_raster.GetGeoTransform(), layer_raster.GetProjection()])
    layer_data.append(layer_raster.ReadAsArray().transpose())
    del rasterband
    del layer_raster

print("- loaded raster data")

### orchestrate normalization & everything else
### -- not implemented...???
### compute expected cube size
expected_cube_size = []
y_layer_max = cube_res[y_base]
for i in range(len(cube_res)):
    if cube_res[i] == y_layer_max:
        expected_cube_size.append(1)
    else:
        expected_cube_size.append(y_layer_max // cube_res[i] + 1)

print("- computed expected cube sizes: ", expected_cube_size)

### make buffer around smaller data layers
buffer_dist = []
buffer_fill = -999
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

### compute offsets for sample generation
center_offset = []
half_offset = []
for i in range(len(layer_data)):
    center_offset.append(expected_cube_size[i] // 2)
    if expected_cube_size[i] % 2 == 0:
        half_offset.append(0.5)
    else:
        half_offset.append(0)

print("- computed sampling offsets")


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

### get mins and maxs for normalization, plus whether it is actually legal
def roundup_layer_1(k, base_idx, buffer_ignore, crs_list, yloc, nd_vals):
    bi, bj = base_idx
    geo_ctr = idx_geo(bi + 0.5, bj + 0.5, crs_list[yloc])
    if k == yloc:
        if layer_data[k][bi, bj] != nd_vals[k] and not np.isnan(layer_data[k][bi, bj]).any():
            return True
    elif expected_cube_size[k] == 1:
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        #print(tidi, tidj)
        if layer_data[k][int(tidi)+buffer_dist[k], int(tidj)+buffer_dist[k]] != nd_vals[k] and \
            not np.isnan(layer_data[k][int(tidi)+buffer_dist[k], int(tidj)+buffer_dist[k]]).any():
            return True
    else:
        ### need to determine UL
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        tidi += buffer_dist[k]
        tidj += buffer_dist[k]
        sulx = int(tidi + half_offset[k]) - center_offset[k]
        suly = int(tidj + half_offset[k]) - center_offset[k]
        temp = layer_data[k][sulx:sulx+expected_cube_size[k],
                             suly:suly+expected_cube_size[k]].reshape(-1)
        if nd_vals[k] not in temp and not np.isnan(temp).any():
            temp = temp[temp != buffer_ignore]
            if len(temp) > 0:
                return True
    return False

### gather legal samples
### ...batch based on regions?
legal_sample_idx_list = []
guide_shape = layer_data[y_base].shape
for i in range(guide_shape[0]):
    if i % (guide_shape[0] // 10) == 0:
        print("-10")
    if i == guide_shape[0] - 1:
        print("last")
    for j in range(guide_shape[1]):
        ### determine if nodata value is involved, and ignore buffer fill values
        all_ok = True
        tmins = []
        tmaxs = []
        for k in range(len(cube_res)):
            ### check individual layer for nodata
            layer_k_np = roundup_layer_1(k, (i, j), buffer_fill, layer_crs, y_base, layer_nodata)
            if layer_k_np == False:
                all_ok = False
                break
        if all_ok:
            legal_sample_idx_list.append((i, j))

print("- rounded up layers: ", len(legal_sample_idx_list))
print("- maximum = ", guide_shape[0] * guide_shape[1])

legal_idx_save = np.array(legal_sample_idx_list)
np.savetxt(fold_name + "/legal_ids.csv", legal_idx_save, delimiter=",")

print("- saved sample coords")

### now do train/test/val splits
meta_indices = np.arange(len(legal_sample_idx_list))
test_indices = []
val_fold_indices = []
train_fold_indices = []
#idea: for each, 0 is unavail, 1 is avail, 2 is taken
if split_method == "blocks":
    n_test_samples = int(len(legal_sample_idx_list) * partition[0])
    approx_each = n_test_samples/split_blocks_nregions
    temp_each_sqrt = math.ceil(math.sqrt(approx_each))
    ### calculate block sizes from parameters
    minerr = 100000
    minoff = 0

    for i in range(max(temp_each_sqrt//5, 1)):
        err1 = abs((math.ceil(approx_each / (temp_each_sqrt - (i - (temp_each_sqrt//10)))) * temp_each_sqrt) - approx_each)
        if err1 < minerr:
            minerr = err1
            minoff = i - (temp_each_sqrt//10)
    blocksizes = (temp_each_sqrt - minoff, math.ceil(approx_each / (temp_each_sqrt - minoff)))
    underest = (blocksizes[0] * blocksizes[1]) - (n_test_samples/split_blocks_nregions)
    intoffset = underest - int(underest)
    n_val_samples = int(len(legal_sample_idx_list) * partition[1])
    approx_val_each = n_val_samples / split_blocks_nregions
    each_err_val = approx_val_each - int(approx_val_each)

    n_test_samples = int(len(legal_sample_idx_list) * partition[0])
    approx_test_each = n_test_samples / split_blocks_nregions
    each_err_test = approx_test_each - int(approx_test_each)

    print("- computed test set block sizes")
    print("  - diagnostics:     total_blocks:", n_test_samples, "approx. each:", approx_each, "sqrt:", temp_each_sqrt)
    print("  -                  blocksizes:", blocksizes, "block each:", blocksizes[0] * blocksizes[1], "est:", underest)
    print("  -                  guid_shape", guide_shape)

    ### make mask grid
    block_mask = np.zeros(guide_shape)
    metaindex_arr = np.zeros(guide_shape)
    for i in range(len(legal_sample_idx_list)):
        ti, tj = legal_sample_idx_list[i]
        block_mask[ti, tj] = 1
        metaindex_arr[ti, tj] = i

    ### place blocks over region
    nplaced = 0
    test_selected = []
    carrier = 0
    while nplaced < split_blocks_nregions:
        ruli = int(np.random.uniform(0, guide_shape[0] - blocksizes[0]))
        rulj = int(np.random.uniform(0, guide_shape[1] - blocksizes[1]))
        overlap = np.where(block_mask[max(ruli-split_blocks_buffer, 0): min(ruli+blocksizes[0] + split_blocks_buffer, guide_shape[0]), max(rulj-split_blocks_buffer, 0): min(rulj+blocksizes[1] + split_blocks_buffer, guide_shape[1])] == 2)
        if len(overlap[0]) == 0 and block_mask[ruli, rulj] == 1:
            if nplaced % (split_blocks_nregions // 10) == 0:
                print("-10")
            nplaced += 1
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

    overlap3 = np.where(block_mask == 2)
    test_indices = metaindex_arr[overlap3]
    remaining_indices = metaindex_arr[np.where(block_mask == 1)]
    print("- found test indices:", len(test_indices), "out of", n_test_samples, "desired")

    ### make tifs for visual confirmation
    os.system("rm " + fold_name + "/fold_box_geotifs/*")
    os.system("mkdir " + fold_name + "/fold_box_geotifs")
    raster_helpers.save_raster(fold_name + "/fold_box_geotifs", "test_extent", block_mask, layer_proj[y_base][0], layer_proj[y_base][1], -1)

    block_mask[overlap3] = 0

    ### now do xval-splits
    for ii in range(n_splits):
        split_mask_i = block_mask.copy()
        ### place blocks over region
        nplaced_i = 0
        carrier = 0
        while nplaced_i < split_blocks_nregions:
            ruli = int(np.random.uniform(split_outer_buffer, guide_shape[0] - split_outer_buffer))
            rulj = int(np.random.uniform(split_outer_buffer, guide_shape[1] - split_outer_buffer))
            overlap = np.where(block_mask[max(ruli - split_blocks_buffer, 0): min(ruli + blocksizes[0] + split_blocks_buffer, guide_shape[0]),
                               max(rulj - split_blocks_buffer, 0): min(rulj + blocksizes[1] + split_blocks_buffer, guide_shape[1])] == 2)
            if len(overlap[0]) == 0 and split_mask_i[ruli, rulj] == 1:
                nplaced_i += 1
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
        overlap4 = np.where(split_mask_i == 2)
        val_fold_indices.append(metaindex_arr[overlap4])
        train_fold_indices.append(metaindex_arr[np.where(split_mask_i == 1)])
        np.savetxt(fold_name + "/train_" + str(ii) + ".csv", train_fold_indices[-1], delimiter=",")
        np.savetxt(fold_name + "/val_" + str(ii) + ".csv", val_fold_indices[-1], delimiter=",")

        ### make geotif for visual inspection
        raster_helpers.save_raster(fold_name + "/fold_box_geotifs", "val_extent_" + str(ii), split_mask_i, layer_proj[y_base][0], layer_proj[y_base][1], -1)

    np.savetxt(fold_name + "/test.csv", test_indices, delimiter=",")
    np.savetxt(fold_name + "/remaining.csv", remaining_indices, delimiter=",")


elif split_method == "fullrand":
    np.random.shuffle(meta_indices)
    test_indices = meta_indices[:int(len(meta_indices) * partition[0])]
    remaining_indices = meta_indices[int(len(meta_indices) * partition[0]):]
    np.random.shuffle(remaining_indices)
    for i in range(n_splits):
        np.random.shuffle(remaining_indices)
        val_fold_indices.append(remaining_indices[:int(len(remaining_indices)*partition[1])])
        train_fold_indices.append(remaining_indices[int(len(remaining_indices)*partition[1]):])
        np.savetxt(fold_name + "/train_" + str(i) + ".csv", train_fold_indices[-1], delimiter=",")
        np.savetxt(fold_name + "/val_" + str(i) + ".csv", val_fold_indices[-1], delimiter=",")

    np.savetxt(fold_name + "/test.csv", test_indices, delimiter=",")
    np.savetxt(fold_name + "/remaining.csv", remaining_indices, delimiter=",")

print("- performed and saved train/val/test splits")

### setup for h5
h5_data_files = []
h5_file_sets = []
h5_chunks = []
h5_chunk_counter = 0
h5_chunkid = 0
for i in range(len(cube_res)):
    os.system("rm " + fold_name + "/layer_"+str(i)+".h5")
    h5_data_files.append(h5py.File(fold_name + "/layer_"+str(i)+".h5", "a"))
    h5_file_sets.append(h5_data_files[i].create_dataset("data", 
                        (h5_chunk_size, expected_cube_size[i], expected_cube_size[i]),
                        maxshape=(None, expected_cube_size[i], expected_cube_size[i]),
                        chunks=(h5_chunk_size, expected_cube_size[i], expected_cube_size[i])))
    h5_chunks.append(np.zeros((h5_chunk_size, expected_cube_size[i], expected_cube_size[i])))
    h5_chunks[-1].fill(-1)

print("- set up h5 datasets")

### should always return a 2d np array...
def roundup_layer_2(k, base_idx, crs_list, yloc):
    bi, bj = base_idx
    geo_ctr = idx_geo(bi + 0.5, bj + 0.5, crs_list[yloc])
    if k == yloc:
        return layer_data[k][[[bi]], [[bj]]]
    elif expected_cube_size[k] == 1:
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        return layer_data[k][[[int(tidi)+buffer_dist[k]]],[[int(tidj)+buffer_dist[k]]]]
    else:
        ### need to determine UL
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        tidi += buffer_dist[k]
        tidj += buffer_dist[k]
        sulx = int(tidi + half_offset[k]) - center_offset[k]
        suly = int(tidj + half_offset[k]) - center_offset[k]
        return layer_data[k][sulx:sulx+expected_cube_size[k],
                             suly:suly+expected_cube_size[k]]

### for folds, organized as [fold][layer]
samples_mins = []
samples_maxs = []
folds_mins = []
folds_maxs = []
for j in range(n_splits):
    folds_mins.append([])
    folds_maxs.append([])
    for i in range(len(layer_data)):
        folds_mins[j].append(float("inf"))
        folds_maxs[j].append(float("-inf"))

for i in range(len(layer_data)):
    samples_mins.append(float("inf"))
    samples_maxs.append(float("-inf"))
        

### go back through combined list and compute min/max
### convert to h5 and save
for i in range(len(meta_indices)):
    ### get data for h5
    ### if not in test, get min/max for normalization
    for k in range(len(cube_res)):
        result = roundup_layer_2(k, legal_sample_idx_list[i], layer_crs, y_base)
        ### add to h5
        h5_chunks[k][h5_chunk_counter%h5_chunk_size, :, :] = result
        if i not in test_indices:
            ### get scaling info
            mmcheck = result[result != buffer_fill]
            if len(mmcheck > 0):
                nanmin = np.nanmin(mmcheck)
                nanmax = np.nanmax(mmcheck)
                ### adjust min/max for combined data
                samples_mins[k] = min(nanmin, samples_mins[k])   
                samples_maxs[k] = max(nanmax, samples_maxs[k])
                ### adjust min/max for individual splits
                for j in range(n_splits):
                    if i in train_fold_indices[j]:
                        folds_mins[j][k] = min(nanmin, folds_mins[j][k])
                        folds_maxs[j][k] = max(nanmax, folds_maxs[j][k]) 

    h5_chunk_counter += 1
    if h5_chunk_counter % h5_chunk_size == 0:
        for k in range(len(cube_res)):
            ### resize to current number of items
            h5_file_sets[k].resize(h5_chunk_counter, axis=0)
            ### set values from current chunk
            h5_file_sets[k][h5_chunk_counter-h5_chunk_size:h5_chunk_counter,:,:] =\
                 np.array(h5_chunks[k][:,:,:])
            h5_chunks[k] = np.zeros(h5_chunks[k].shape)
            h5_chunks[k].fill(-1)

print("- reformatted data to h5")

### finish remainder of h5 chunk and save
for k in range(len(cube_res)):
    h5_file_sets[k].resize(h5_chunk_counter, axis=0)
    h5_file_sets[k][h5_chunk_counter-(h5_chunk_counter%h5_chunk_size):h5_chunk_counter,:,:] =\
         np.array(h5_chunks[k][:h5_chunk_counter%h5_chunk_size,:,:])
    h5_data_files[k].close()

print("- saved h5 files")
print("- calculated min/max normalization parameters")
combined_mins = np.array(samples_mins)
combined_maxs = np.array(samples_maxs)
np.savetxt(fold_name + "/norm_layer_mins_combined.csv", combined_mins, delimiter=",")
np.savetxt(fold_name + "/norm_layer_maxs_combined.csv", combined_maxs, delimiter=",")

for i in range(n_splits):
    fold_np_mins = np.array(folds_mins[i])
    fold_np_maxs = np.array(folds_maxs[i])
    np.savetxt(fold_name + "/norm_layer_mins_fold_"+str(i)+".csv", fold_np_mins, delimiter=",")
    np.savetxt(fold_name + "/norm_layer_maxs_fold_"+str(i)+".csv", fold_np_maxs, delimiter=",")

print("- saved min/max normalization parameters")
print("- done")


"""

            if tlr == 0 and ttb == 0:
                if tr == 0:
                    iupdate = 1
                else:
                    jupdate = 1
            if tlr == 0 and ttb == 1:
                if tr == 0:
                    jupdate = -1
                else:
                    iupdate = 1
            if tlr == 1 and ttb == 1:
                if tr == 0:
                    iupdate = -1
                else:
                    jupdate = -1
            if tlr == 1 and ttb == 0:
                if tr == 0:
                    jupdate = 1
                else:
                    iupdate = -1
            tlr = (tlr * (blocksizes[0]-1)) + ruli
            ttb = (ttb * (blocksizes[1]-1)) + rulj
            if leaveoff > 0:
                tlo = leaveoff
                while tlo > 0:
                    if block_mask[tlr, ttb] == 2:
                        block_mask[tlr, ttb] = 1
                        tlo -= 1
                    ### update pos... need a much better way to do this...
                    if tlr + iupdate >= ibds[0][0] and tlr + iupdate < ibds[0][1] and ttb + jupdate >= ibds[1][0] and ttb + jupdate < ibds[1][1]:
                        pass
                    elif tlr + iupdate < ibds[0][0]:
                        if tr == 0:
                            iupdate = 0
                            jupdate = -1
                            ibds[1][1] -= 1
                        else:
                            iupdate = 0
                            jupdate = 1
                            ibds[1][0] += 1
                    elif tlr + iupdate >= ibds[0][1]:
                        if tr == 0:
                            iupdate = 0
                            jupdate = 1
                            ibds[1][0] += 1
                        else:
                            iupdate = 0
                            jupdate = -1
                            ibds[1][1] -= 1
                    elif ttb + jupdate < ibds[1][0]:
                        if tr == 0:
                            iupdate = 1
                            jupdate = 0
                            ibds[0][0] += 1
                        else:
                            iupdate = -1
                            jupdate = 0
                            ibds[0][1] -= 1
                    elif ttb + jupdate < ibds[1][1]:
                        if tr == 0:
                            iupdate = -1
                            jupdate = 0
                            ibds[0][1] -= 1
                        else:
                            iupdate = 1
                            jupdate = 0
                            ibds[0][0] += 1
                    tlr += iupdate
                    ttb += jupdate

            elif leaveoff < 0:
                print("error... leaveoff < 0")
                pass"""

"""overlap2 = np.where(block_mask[ruli: ruli+blocksizes[0], rulj:rulj+blocksizes[1]] == 1)
            if len(overlap2[0]) > ((blocksizes[0] * blocksizes[1]) - underest):
                nplaced += 1
                block_mask[overlap2] = 2
                ### do occlusion
                carrier += intoffset
                leaveoff = int(underest) - ((blocksizes[0] * blocksizes[1]) - len(overlap2[0])) + int(carrier)
                print(underest, blocksizes[0], blocksizes[1], len(overlap2[0]), int(carrier))
                print("base leaveoff:", leaveoff)
                carrier = carrier % 1
                tlr = int(np.random.uniform(0, 2))
                ttb = int(np.random.uniform(0, 2))
                tr = int(np.random.uniform(0, 2))
                ibds = np.array([ruli, ruli+blocksizes[0], rulj, rulj+blocksizes[1]])
                ### see stupid matrix math in small notebook for explanation of what is going on here
                ijvec = np.array([[tr * (1 - tlr - ttb) + (1-tr) * (ttb - tlr)], [tr * (tlr - ttb) + (1-tr) * (1 - tlr - ttb)]])
                rmat = np.array([[0, 1 - (2*tr)], [(2*tr) - 1, 0]])

                tlr = int((tlr * (blocksizes[0]-1)) + ruli)
                ttb = int((ttb * (blocksizes[1]-1)) + rulj)
                if leaveoff > 0:
                    tlo = leaveoff
                    while tlo > 0:
                        if block_mask[tlr, ttb] == 2:
                            block_mask[tlr, ttb] = 1
                            tlo -= 1
                        if tlr + ijvec[0] < ibds[0][0] or tlr + ijvec[0] >= ibds[0][1] or ttb + ijvec[1] < ibds[1][0] or ttb + ijvec[1] >= ibds[1][1]:
                            ### apply crop
                            ### see equations in small notebook for explanation of what is going on here
                            crop = np.array([0.5 * ijvec[1] * (1 - ijvec[0]) * (1 + ijvec[1] - 2*tr),
                                    -0.5 * (1-ijvec[0]) * ijvec[1] * (2*tr + ijvec[1] -1),
                                    0.5 * ijvec[0] * (1 - ijvec[1]) * (2*tr + ijvec[0] - 1),
                                    -0.5 * ijvec[0] * (1 - ijvec[1]) * (ijvec[0] + 1 - 2*tr)])
                            ibds += crop
                            ### apply rotate
                            ijvec = rmat @ ijvec
                        ### apply transform
                        tlr += int(ijvec[0])
                        ttb += int(ijvec[1])

                elif leaveoff < 0:
                    print("error at step", nplaced, "... leaveoff < 0 (", leaveoff, ")")
                    pass"""