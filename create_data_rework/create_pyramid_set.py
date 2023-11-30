### create_pyramid_set.py
if __name__ == "__main__":
    from osgeo import gdal
    import numpy as np
    import h5py
    import os
    import sys
    import math
    import raster_helpers
    import create_pyramid_functions as cpf
    gdal.UseExceptions()

    print("- loaded packages")

    ### Start from scratch or load from checkpoint
    load_checkpoint = False

    ### USER PARAMETERS
    ### Required CRS: EPSG 4326 WSG 84
    data_input_crs = "EPSG4326WSG84"

    ### location, base resolution, sample resolution, x/y, idx
    data_info = [["../data/raster/srtm_clipped_co.tif",                 30,     30,     "x", 0,     "srtm"],
                 ["../data/raster/nlcd_clipped_co_reproj.tif",          30,     30,     "x", 8,     "nlcd2019"],
                 ["../data/raster/aspect_clipped_co.tif",               30,     30,     "x", 10,    "aspect"],
                 ["../data/raster/slope_clipped_co.tif", 30, 30, "x", 11, "slope"],
                 ["../data/raster/treeage_clipped_co_reproj.tif", 1000, 1000, "x", 12, "treeage"],
                 ["../data/raster/ecostresswue_clipped_co.tif",         70,     70,     "y", 19,    "ecostresswue"],
                 ["../data/raster/ecostressesi_clipped_co.tif",         70,     70,     "y", 20,    "ecostressesi"],
                 ["../data/raster/gedi_agforestbiomass_clipped_co.tif", 1000,   1000,   "y", 21,    "gediagb"]]
    """data_info = [["../data/raster/srtm_clipped_co.tif",                 30,     30,     "x", 0,     "srtm"],
                 ["../data/raster/nlcd2001_clipped_co_reproj.tif",      30,     30,     "x", 1,     "nlcd2001"],
                 ["../data/raster/nlcd2004_clipped_co_reproj.tif",      30,     30,     "x", 2,     "nlcd2004"],
                 ["../data/raster/nlcd2006_clipped_co_reproj.tif",      30,     30,     "x", 3,     "nlcd2006"],
                 ["../data/raster/nlcd2008_clipped_co_reproj.tif",      30,     30,     "x", 4,     "nlcd2008"],
                 ["../data/raster/nlcd2011_clipped_co_reproj.tif",      30,     30,     "x", 5,     "nlcd2011"],
                 ["../data/raster/nlcd2013_clipped_co_reproj.tif",      30,     30,     "x", 6,     "nlcd2013"],
                 ["../data/raster/nlcd2016_clipped_co_reproj.tif",      30,     30,     "x", 7,     "nlcd2016"],
                 ["../data/raster/nlcd_clipped_co_reproj.tif",          30,     30,     "x", 8,     "nlcd2019"],
                 ["../data/raster/nlcd2021_clipped_co_reproj.tif",      30,     30,     "x", 9,     "nlcd2021"],
                 ["../data/raster/aspect_clipped_co.tif",               30,     30,     "x", 10,    "aspect"],
                 ["../data/raster/slope_clipped_co.tif",                30,     30,     "x", 11,    "slope"],
                 ["../data/raster/treeage_clipped_co_reproj.tif",       1000,   1000,     "x", 12,    "treeage"],
                 ["../data/raster/prism_precip_30y_800m_reproj.tif",    800,    800,     "x", 13,    "precip"],
                 ["../data/raster/prism_tempmin_30y_800m_reproj.tif",   800,    800,     "x", 14,    "tempmin"],
                 ["../data/raster/prism_tempmean_30y_800m_reproj.tif",  800,    800,     "x", 15,    "tempmean"],
                 ["../data/raster/prism_tempmax_30y_800m_reproj.tif",   800,    800,     "x", 16,    "tempmax"],
                 ["../data/raster/prism_vapormin_30y_800m_reproj.tif",  800,    800,     "x", 17,    "vapormin"],
                 ["../data/raster/prism_vapormax_30y_800m_reproj.tif",  800,    800,     "x", 18,    "vapormax"],
                 ["../data/raster/ecostresswue_clipped_co.tif",         70,     70,     "y", 19,    "ecostresswue"],
                 ["../data/raster/ecostressesi_clipped_co.tif",         70,     70,     "y", 20,    "ecostressesi"],
                 ["../data/raster/gedi_agforestbiomass_clipped_co.tif", 1000,   1000,   "y", 21,    "gediagb"]]"""

    ### split method and parameters
    y_base = len(data_info) - 1
    # how to divide geographic areas for train/val/test split
    split_method = "blocks" # in {blocks, fullrand}
    split_blocks_buffer = 10
    split_blocks_nregions = 100
    split_outer_buffer = 2

    ### partition parameters
    n_splits = 3
    partition = (0.3, 0.2)

    ### input data parameters
    exclude = []
    run_name = "mf_test"
    fold_name = "../data/pyramid_sets/" + run_name

    print('- creating fold "' + run_name +'"')

    ### data processing parameters
    buffer_fill = -999
    np_random_seed = 100807
    h5_chunk_size = 1000
    parallelize = True
    reduce_to_total_dims = False
    np.random.seed(np_random_seed)
    print("- set random seed to", np_random_seed)

    ### auto params
    layer_locs = []
    cube_res = []
    sample_to_res = []
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
            sample_to_res.append(data_info[i][2])
            if data_info[i][3] == "x":
                x_layers.append(i)
            else:
                y_layers.append(i)

    checkpoint_prev = -1
    if load_checkpoint:
        ###
        try:
            checkpoint_prev = cpf.get_checkpoint(fold_name)
            print("- previous checkpoint found:", checkpoint_prev)
        except:
            print("- error loading checkpoint, setting checkpoint to 0")
            checkpoint_prev = 0

    ### checkpoint 0
    if checkpoint_prev <= 0:
        os.system("mkdir " + fold_name)
        cpf.set_checkpoint(fold_name, checkpoint_number=0)
        print("- created base directory")

    ### LOAD DATA
    ### need to do regardless of checkpoint
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

    print("- ndvals:", layer_nodata)
    ### ORCHESTRATE

    ### compute expected cube size (expected size of each layer in pyramid)
    ### need to compute regardless of checkpoint
    expected_cube_size = cpf.compute_expected_cube_sizes(cube_res[y_base], cube_res)
    expected_sample_to = cpf.compute_expected_cube_sizes(cube_res[y_base], sample_to_res)
    if reduce_to_total_dims:
        expected_sample_to = cpf.reduce_from_total_dims(expected_cube_size, expected_sample_to, x_layers)
    ### save info file
    cpf.save_info_file(data_info, expected_sample_to, fold_name, n_splits, buffer_fill, data_input_crs, np_random_seed)
    ### make a buffer around data layers to avoid going out of bounds... involves resizing data
    buffer_dist, layer_data = cpf.make_buffer(buffer_fill, layer_data, cube_res, y_base)
    ### compute sampling offsets for dealing with odd and even sizes in sample generation
    center_offset, half_offset = cpf.compute_offsets(expected_cube_size, layer_data)

    ### roundup 1: compile list of legal samples
    ### checkpoint 1
    if checkpoint_prev <= 1:
        cpf.set_checkpoint(fold_name, checkpoint_number=1)
        legal_sample_idx_list, guide_shape = cpf.compile_legal_samples(expected_cube_size, layer_data, y_base,
                                                                       cube_res, buffer_fill, layer_crs,
                                                                       layer_nodata, buffer_dist, half_offset,
                                                                       center_offset)
        ### save legal sample index list
        cpf.save_legal_sample_ids(legal_sample_idx_list, fold_name)
    ### if already computed, load...
    else:
        legal_sample_idx_list = cpf.load_legal_sample_ids(fold_name)
        guide_shape = layer_data[y_base].shape

    ### now do train/test/val splits
    ### checkpoint 2
    if checkpoint_prev <= 2:
        cpf.set_checkpoint(fold_name, checkpoint_number=2)
        meta_indices = np.arange(len(legal_sample_idx_list))
        if split_method == "blocks":
            test_indices, remaining_indices, train_fold_indices, val_fold_indices = cpf.block_split(legal_sample_idx_list, partition, split_blocks_nregions, guide_shape, split_blocks_buffer, fold_name, layer_proj, y_base, n_splits, split_outer_buffer)
        elif split_method == "fullrandom":
            test_indices, remaining_indices, train_fold_indices, val_fold_indices = cpf.fullrand_split(legal_sample_idx_list, partition, n_splits, fold_name)
    else:
        ### load indices
        test_indices, remaining_indices, train_fold_indices, val_fold_indices = cpf.load_splits(fold_name, n_splits)

    ### parallelize pyramid setup
    ### checkpoint 3
    if checkpoint_prev <= 3:
        cpf.set_checkpoint(fold_name, checkpoint_number=3)
        if parallelize:
            cpf.make_pyramids_main(cube_res, fold_name, h5_chunk_size, expected_cube_size,
                               legal_sample_idx_list, layer_crs, y_base, test_indices, buffer_fill, n_splits,
                               layer_data, train_fold_indices, expected_sample_to, buffer_dist, half_offset,
                               center_offset)
        else:
            cpf.pyramid_nonparallel(fold_name, h5_chunk_size, expected_cube_size, n_splits, layer_data, meta_indices,
                                    cube_res, legal_sample_idx_list, layer_crs, y_base, test_indices,
                                    train_fold_indices, buffer_fill, sample_to_res, buffer_dist, half_offset,
                                    center_offset)
    else:
        print("- all done without reaching checkpoint")

    ### checkpoint 4
    cpf.set_checkpoint(fold_name, checkpoint_number=4)



