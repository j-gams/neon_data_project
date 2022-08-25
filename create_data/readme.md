# PIPELINE TO CREATE DATA SET
## Run the entire data creation pipeline with create_data.sh
This does the following, with certain default settings:
- tif_merge_convert.py
  (the srtm data comes in multiples smaller geotifs. This merges them into one combined geotif.)
- code_match.py
  (the raster data is not clipped to the aoi. This clips all raster input data to the aoi.)
- analyze_clipped.py
  (This plots the distribution of raster values for each raster data type at each gedi centroid.)
- check_clip.py
  (This plots the clipped raster data. Make sure everything looks ok.)
- reset_raster_nd.py
  (The srtm data might have an incorrect no-data value. This allows for checking no-data (nd) values and correcting them if any are not 'nan'.)
- testbed.py
  (This tests stuff. At present, it prints out data field names for gedi data.)
- match_create_set.py
  (This is the monstrous script that creates datasets. Check the default settings to make sure that it is doing the right thing!)
- test_xdata.py
  (This plots channels of x samples as images as a sanity check for match_create_set.py.)
- build_train_val_test.py
  (This creates train/validation/test splits on already created datasets. Check the default settings for this as well!)
- h5_sanitycheck
  (This compares samples from an h5-structured dataset to those from a csv-structured dataset to make sure that the samples have the same values.)
  
## Documentation
### tif_merge_convert.py
### code_match.py
### analyze_clipped.py
### check_clip.py
### reset_raster_nd.py
### match_create_set.py
#### Parameters
| Parameter | Usage | Function | Example usage |
| --- | --- | --- | --- |
| X raster data path(s) | (required) comma separated file paths | Which clipped raster data to include in the unified dataset | ../raw_data/srtm_raw/srtm_clipped.tif,../raw_data/nlcd_raw/nlcd_clipped.tif |
| point-interpolated data path(s) | (required) comma separated file paths | which point shapefiles to include in the unified dataset | ../raw_data/gedi_pts/GEDI_2B_clean.shp |
| y raster data path | (required) file path | raster data to use as y value for datacube samples | ../raw_data/ecos_wue/wue_median_composite_clipped.tif |
| y resolution | (required) int | resolution in meters of y raster data. Used to determine the size of datacubes with y pixel size. | 70 |
| y pixel size | (required) int | resolution in meters of output datacubes | 5 |
| create file structure | (required) {T, t, True, true, F, f, False, false} | whether to try to create a new file structure for the dataset or assume one already exists (this should be True when first creating a dataset and False when revising one) | True |
| dataset path | (required) path for root directory of the dataset | ../data/data_h51 |
| low-memory mode | (optional, default false) --lomem to run in low memory mode | whether to be cautious about loading a lot of data into memory at once. If in low-memory mode the program will rewrite relevant fields (specified by the critical fields argument) in a way that reduces memory required but will take longer, if generate coordinates and /or generate other data are set to true. Every field containing the same keyword will be written to the same file. Again, this will take a long time. If low memory mode is on but generating fields is not requested, it will attempt to load precomputed individual files. | --lomem |
| generate coordinates | (optional, default false) --gencoords to reformat gedi coordinate data | This reformats gedi coordinate data in a more memory-friendly way (by saving to np array instead of keeping in shapefile) that also reduces future overhead when revising the dataset | --gencoords |
| generate other data | (optional, default false) --genetc to reformat critical field data | This reformats all fields containing keywords specified in the critical fields parameter in a more memory-friendly way that also reduces future overhead when revising the dataset | --genetc |
| override restructuring | (optional, default false) --override to delete and replace any preexisting reformatted data files. | When run in low memory mode without override, the script will attempt to load files even if --gencoords and --genetc are toggled. When run with override, the files will be ignored and recomputed. | --override | 
| skip save | (optional, default false) --skipsave to generate samples but not save them | This is useful when debugging code, but shouldn't be used otherwise. | --skipsave |
| hdf5 mode | (optional, default false) --h5mode to run in hdf5 mode | This will save samples in one large h5 file instead of individual csv files. This is recommended in most cases because it is much faster for machine learning. The drawback to using h5 files is they cannot be visually inspected as easily as  csv files. | --h5mode |
| hdf5 and csv mode | (optional, default false) --h5both to save to h5 and csv files | This will save the same samples to an h5 dataset and csv files with the same indices. This is required to run h5_sanitycheck.py |
| no shuffle mode | (optional, default false) --noshuffle to turn off index shuffling | When creating unified samples, the program iterates through every grid square in the y raster. By default, the order in which the array is traversed is randomized (it iterates through shuffled lists of x and y indices). This option will prevent the indices from being shuffled, so the order of traversal will be standard. | --noshuffle |
| prescreen | (optional, default true) --prescreen to ignore extreme samples | This makes the program ignore samples that contain extreme values (values greater in absolute value than 10^5). This addresses some issues in the slope and aspect data. This setting is recommended. | --prescreen |
| critical fields | (optional, default ???) -c comma separated field keywords | This is not exactly optional as it specifies which data fields from the GEDI data should be included in the unified data. To include all fields that include "cover", use cover as a keyword. To only use "cover_z_11", use that as a keyword. | -c cover,pavd,fhd |
| k closest approximation | (optional, default 10) -k int to set k closest approximation | This is a deprecated parameter for an algorithm used to speed up the nearest-neighbor interpolation used on GEDI data. It is best to leave this alone. | -k 20 |
| test mode | (optional, default -1) -t int to run in test mode | This creates only a subset of the dataset when provided an integer over 0 and the entire dataset when provided an integer below 0. | -t 50000 |
| channel mode | (optional, default 'hwc') -m {hwc, chw} to set channel mode | This determines whether to format unified data in height,width,channel mode or channel,height,width mode. hwc mode is recommended for most if not all machine learning models within the scope of this project. | -m hwc | 
| pad image | (optional, default 0) -p int to set image pad | This is used to nudge the dimensions of the output samples. Ecostress is 70m resolution, GEDI is 25m, and the other raster data is 30m. If we want the resolution of the output to be 5m (the greatest common factor of these) then the output is 14x14, which is an awkward dimension. Padding 1 pixel allows for a much nicer 16x16. To achieve this, data is sampled from a 16x16 grid at 5m resolution centered on the 70x70m raster square instead of a 14x14 grid at 5m resolution centered on the 70m raster square. This slightly larger datacube is still associated with the 'y' value of the central ecostress square. In general, a padding value of n will result in a 14+2n x 14+2n sample. | -p 1 |
| pad hash | (optional, default 1) -h int to pad the nearest neighbor hash | This is a parameter for part of an algorithm used to speed up the nearest-neighbor interpolation used on GEDI data. The recommended hash pad value is 10. Too small a number will lead to incorrect interpolation or broken code, and too high a number will lead to increased memory usage. | -h 10 |
| h5 chunk size |  (optional, default 1000) -u int to set chunk size | This determines the chunk size for h5 files, if in use. Larger numbers may speed up computation slightly but result in higher memory usage. 1000 - 10000 recommended. | -u 1000 |
| verbosity | -q {0, 1, 2} to set verbosity | This is largely not yet implemented | -q 1 |
#### Recommended commands
First time use (h5 mode, entire dataset):
```
python match_create_set.py ../raw_data/srtm_raw/srtm_clipped.tif,../raw_data/nlcd_raw/nlcd_clipped.tif,../raw_data/slope_raw/slope_clipped.tif,../raw_data/aspct_raw/aspct_clipped.tif ../raw_data/gedi_pts/GEDI_2B_clean.shp ../raw_data/ecos_wue/wue_median_composite_clipped.tif 70 5 true ../data/data_h51 --lomem --gencoords --genetc --override --h5mode --prescreen -c cover,pavd,fhd --m hwc -p 1 -h 10 -u 1000 -q 2
```

Remaking (revising) a pre-existing dataset:
```
python match_create_set.py ../raw_data/srtm_raw/srtm_clipped.tif,../raw_data/nlcd_raw/nlcd_clipped.tif,../raw_data/slope_raw/slope_clipped.tif,../raw_data/aspct_raw/aspct_clipped.tif ../raw_data/gedi_pts/GEDI_2B_clean.shp ../raw_data/ecos_wue/wue_median_composite_clipped.tif 70 5 true ../data/data_h51 --lomem --gencoords --genetc --h5mode --prescreen -c cover,pavd,fhd --m hwc -p 1 -h 10 -u 1000 -q 2
```

#### Under the hood
### build_train_val_test.py
### datacube_set.py
### dat_obj.py
### h5_sanitycheck.py
