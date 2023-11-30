### create_split.py

from osgeo import gdal
import numpy as np
import h5py
import os
import sys

print("- loaded packages")

### EPSG 4326 WSG 84

### user parameters
layer_locs = ["../data/geotifs_sampled_1/srtm_30_globalUL.tif",
              "../data/geotifs_sampled_1/nlcd_30_globalUL.tif",
              "../data/geotifs_sampled_1/aspect_30_globalUL.tif",
              "../data/geotifs_sampled_1/slope_30_globalUL.tif",
              "../data/geotifs_sampled_1/ecostressesi_clipped_co.tif"]
n_splits = 1
partition = (0.3, 0.2)
cube_res = [30, 30, 30, 30, 70]
y_base = 4
y_layers = [4]
x_layers = [0, 1, 2, 3]
np_random_seed = 100807
fold_name = "../data/sr_1"
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

big_x = []
big_y = []

n_dumb_samples = (500, 500)
sample_offset = (200, 200)

#21x21... 630m2
#210 meters by 210 meters... so 3x3 esi, 7x7 srtm

for i in range(4):
    big_x.append(np.zeros((40000, 21, 21)))
big_y = np.zeros((40000,9,9))

for i in range(200):
    for j in range(200):
        #sample...
        uli = (20 * 630) + 630*i
        ulj = (20 * 630) + 630*j
        for k in range(4):
            big_x[k][i*200 + j, :, :] = layer_data[k][uli//30: uli//30 + 21, ulj//30: ulj//30 + 21]
        big_y[i*200 + j, :, :] = layer_data[4][uli//70: uli//70 + 9, ulj//70: ulj//70 + 9]

for k in range(4):
    arr_reshaped = big_x[k].reshape(big_x[k].shape[0], -1)
  
    # saving reshaped array to file.
    np.savetxt("bx_"+str(k)+".txt", arr_reshaped)

y_reshaped = big_y.reshape(big_y.shape[0], -1)
np.savetxt("by.txt", y_reshaped)
sys.exit(0)
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

h5_x_size = 9
h5_y_size = 3

subset_num = 10000
### setup for h5
h5_data_files = []
h5_file_sets = []
h5_chunks = []
h5_chunk_counter = 0
h5_chunkid = 0
for i in range(len(4)):
    os.system("rm " + fold_name + "/layer_"+str(i)+".h5")
    h5_data_files.append(h5py.File(fold_name + "/layer_"+str(i)+".h5", "a"))
    h5_file_sets.append(h5_data_files[i].create_dataset("data", 
                        (h5_chunk_size, h5_x_size, h5_x_size),
                        maxshape=(None, h5_x_size, h5_x_size),
                        chunks=(h5_chunk_size, h5_x_size, h5_x_size)))

    h5_data_files.append(h5py.File(fold_name + "/layer_"+str(i)+".h5", "a"))
    h5_file_sets.append(h5_data_files[i].create_dataset("data", 
                        (h5_chunk_size, h5_y_size, h5_y_size),
                        maxshape=(None, h5_y_size, h5_y_size),
                        chunks=(h5_chunk_size, h5_y_size, h5_y_size)))
    h5_chunks.append(np.zeros((h5_chunk_size, h5_y_size, h5_y_size)))
    h5_chunks[-1].fill(-1)

print("- set up h5 datasets")        

### go back through combined list and compute min/max
### convert to h5 and save
for i in range(len(meta_indices)):
    ### get data for h5
    ### if not in test, get min/max for normalization
    for k in range(len(cube_res)):
        result = roundup_layer_2(k, legal_sample_idx_list[i], layer_crs, y_base)
        ### add to h5
        h5_chunks[k][h5_chunk_counter%h5_chunk_size, :, :] = result

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