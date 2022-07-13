# DATA CREATION PIPELINE
# 1. MERGE SRTM TIF FILES
#    - note that the input srtm tifs they need to be in a specific directory structure before you can do this
#    - each file in its own directory within a common directory
python tif_merge_convert.py .hgt --subdirs ../raw_data/srtm_raw

# 2. TRIM DATA TO AOI
python code_match.py ../raw_data/nlcd_raw/reduced_nlcd.tif,../raw_data/srtm_raw/combined.tif,../raw_data/slope_raw/slope_clip1.tif,../raw_data/aspct_raw/Aspect_clip1.tif,../raw_data/ecos_wue/WUE_Median_Composite_AOI.tif ../raw_data/nlcd_raw/nlcd_clipped.tif,../raw_data/srtm_raw/srtm_clipped.tif,../raw_data/slope_raw/slope_clipped.tif,../raw_data/aspct_raw/aspct_clipped.tif,../raw_data/ecos_wue/wue_median_composite_clipped.tif ../raw_data/frontrange_aoi/Front_range.shp -v none -q 2

# 3. ANALYZE CLIPPED DATA
python analyze_clipped.py ../raw_data/srtm_raw/srtm_clipped.tif,../raw_data/nlcd_raw/nlcd_clipped.tif,../raw_data/slope_raw/slope_clipped.tif,../raw_data/aspct_raw/aspct_clipped.tif,../raw_data/ecos_wue/wue_median_composite_clipped.tif ../raw_data/gedi_pts/GEDI_2B_clean.shp -f 0 -t 200 -q 2

# 4. RESET RASTER ND VALUE FOR SRTM
python reset_raster_nd.py ../raw_data/srtm_raw/srtm_clipped.tif nan t

# 5. BUILD DATASET
python match_create_set.py ../raw_data/srtm_raw/srtm_clipped.tif,../raw_data/nlcd_raw/nlcd_clipped.tif,../raw_data/slope_raw/slope_clipped.tif,../raw_data/aspct_raw/aspct_clipped.tif ../raw_data/gedi_pts/GEDI_2B_clean.shp ../raw_data/ecos_wue/wue_median_composite_clipped.tif 70 5 true ../data/data_interpolated --gencoords --genetc --lomem --override -c cover,pavd,fhd -q 2 -t 50000 -p 1 -m hwc -h 10

#6. BUILD SPLITS/FOLDS -- here with one fold
python build_train_val_test.py ../data/data_interpolated test_fold 1 
