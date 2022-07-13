# NEON Data Project

What is on github at present:
- pipeline for creating dataset (create_data/)
  - tif_merge_convert.py (merge multiple geotifs into one raster dataset)
  - code_match.py (trim multiple geotifs to aoi)
  - analyze_clipped.py (plot distribution of values at points of interest)
  - check_clip.py (make sure the clipped data looks ok)
  - reset_raster_nd.py (check and/or reset no-data value of raster)
  - testbed.py (investigate shapefile fields)
  - match_create_set.py (build actual dataset from raster data, shapefile data)
  - test_xdata.py (look at resultant image-style data)
  - build_train_val_test.py (split data, do cross-validation folds)
- dataset handling tools (create_data/)
  - datacube_set.py (handle preprocessing, batches, etc. for one fold of data)
  - dat_obj.py (load data, set up and manage all folds of dataset)
- test models
  - test_model.py (basic cnn model)
- model training framework (not yet complete)
  - train_frame.py (framework for training ML models)
  - model_train.py (manages the training of models over crossval folds, logging, etc)
  - utils.py (utilities for training model)
  - logger.py (log info from the training process)
