# NEON Data Project

## Summary
This project covers building a dataset, training and evlauating machine learning models on the data, and running diagnostics. 
The unified dataset combines Ecostress Water Use Efficiency, SRTM, NLCD (2019), slope, aspect, and GEDI level 2B cover, pavd, and fhd data at a 5m resolution.

## Structure

neon_data_project
 - raw_data
   - aspect_raw
   - ecostress_wue
   - frontrange_aoi
   - gedi_pts
   - nlcd_raw
   - srtm_raw
   - slope_raw
 - data
   - (your datasets)
 - create_data
   - tif_merge_convert.py
   - code_match.py
   - analyze_clipped.py
   - check_clip.py
   - reset_raster_nd.py
   - match_create_set.py
   - rfdata_loader.py
   - build_train_val_test.py
   - h5_sanitycheck.py
   - dat_obj.py
   - datacube_set.py
   - corr_analysis.py
   - testbed.py
   - test_vnoi.py
   - test_xdata.py
   - voronator.py
 - models
   - custom_models
     - cnn_1.py
     - auto_regress.py
     - kernel_regress.py
     - lasso_regress.py
     - reduce_regress.py
     - regress.py
     - rf_regress.py
     - svr_1.py
   - saved_models
     - (auto-saved models)
   - train_frame.py
   - model_train.py
   - mutils.py
   - logger.py
 - figures
   - corr_comparison
     - (auto-generated figures)
   - data
     - (auto-generated figures)
   - gedi_distributions
     - (auto-generated figures)
   - pixel_distributrions
     - (auto-generated figures)
   - (auto-generated figures)
 - logs
   - (auto-generated logs)
