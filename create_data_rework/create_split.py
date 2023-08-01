### create_split.py

from osgeo import gdal
import numpy as np
import h5py
import os

print("- loaded packages")

### EPSG 4326 WSG 84

### user parameters
layer_locs = ["../data/raster/srtm_clipped_co.tif",
              "../data/raster/nlcd_clipped_co_reproj.tif",
              "../data/raster/aspect_clipped_co.tif",
              "../data/raster/slope_clipped_co.tif",
              "../data/raster/treeage_clipped_co_reproj.tif",
              "../data/raster/ecostressesi_clipped_co.tif",
              "../data/raster/gedi_agforestbiomass_clipped_co.tif"]
n_splits = 1
partition = (0.3, 0.2)
cube_res = [30, 30, 30, 30, 1000, 70, 1000]
y_base = 6
y_layers = [5, 6]
x_layers = [0, 1, 2, 3, 4]
np_random_seed = 100807
fold_name = "../data/test_2"
h5_chunk_size = 1000



### auto params
layer_names = []
layer_nodata = []
layer_size = []
layer_crs = []
layer_data = []

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
    layer_data.append(layer_raster.ReadAsArray().transpose())
    del rasterband
    del layer_raster

print("- loaded raster data")

### orchestrate normalization & everything else
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
    half_offset
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
legal_sample_idx_list = []
guide_shape = layer_data[y_base].shape
for i in range(guide_shape[0]):
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
            """else:
                tmins.append(layer_k_np[0])
                tmaxs.append(layer_k_np[1])"""
        if all_ok:
            """
            for k in range(len(tmins)):
                if tmins[k] != buffer_fill:
                    samples_mins[k] = min(samples_mins[k], tmins[k])
                if tmaxs[k] != buffer_fill:
                    samples_maxs[k] = max(samples_maxs[k], tmaxs[k])"""
            legal_sample_idx_list.append((i, j))

print("- rounded up layers: ", len(legal_sample_idx_list))
print("- maximum = ", guide_shape[0] * guide_shape[1])

legal_idx_save = np.array(legal_sample_idx_list)
np.savetxt(fold_name + "/legal_ids.csv", legal_idx_save, delimiter=",")

print("- saved sample coords")

### now do train/test/val splits
meta_indices = np.arange(len(legal_sample_idx_list))
np.random.seed(np_random_seed)
np.random.shuffle(meta_indices)
test_indices = meta_indices[:int(len(meta_indices) * partition[0])]
remaining_indices = meta_indices[int(len(meta_indices) * partition[0]):]
val_fold_indices = []
train_fold_indices = []
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