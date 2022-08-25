# PIPELINE TO CREATE DATA SET
## Run the entire data creation pipeline with create_data.sh
This does the following, with certain default settings:
- tif_merge_convert.py
  (the srtm data comes in multiples smaller geotifs. This merges them into one combined geotif)
- code_match.py
  (the raster data is not clipped to the aoi. This clips all raster input data to the aoi)
- analyze_clipped.py
  (plot distribution of raster values for each raster data type at each gedi centroid)
- check_clip.py
  (plot the clipped raster data. Make sure everything looks ok)
- reset_raster_nd.py
  (the srtm data might have an incorrect no-data value. Check nd values with this and correct if any are not 'nan')
- testbed.py
  (test stuff. At present, prints out data field names for gedi data)
- match_create_set.py
  (the big boy. Feed this all the raster and gedi data, and it will create a dataset)
- test_xdata.py
  (plot x images created by match_create_set.py, for sanity check)
- build_train_val_test.py
  (split data into different sets, cross-validation folds as needed)
