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
| Parameters | Usage | Function | Default (example) |
| --- | --- | --- | --- |
| X raster data path(s) | (required) comma separated file paths | Which clipped raster data to include in the unified dataset | ../raw_data/srtm_raw/srtm_clipped.tif,../raw_data/nlcd_raw/nlcd_clipped.tif |

| point-interpolated data path(s) | (required) comma separated file paths | which point shapefiles to include in the unified dataset | ../raw_data/gedi_pts/GEDI_2B_clean.shp |

| y raster data path | (required) file path | raster data to use as y value for datacube samples | ../raw_data/ecos_wue/wue_median_composite_clipped.tif |
### build_train_val_test.py
### datacube_set.py
### dat_obj.py
### h5_sanitycheck.py
