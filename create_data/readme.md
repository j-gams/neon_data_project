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
This is used to stitch multiple geotifs from the same dataset into one combined geotif.
#### Parameters
| Parameter | Usage | Function |
| --- | --- | --- |
| target extension | required string | The file extension to look for |
| target directories | required string of comma separated directories | the directories to look in |
| subdirectory mode | (optional, default False) --subdirs to engage subdirectory mode | engage this to look for geotifs to stitch together in subdirectories of the target directory, not the directory itself. This is useful, for example, if the constituent geotifs come in .zip files that place the geotifs in individual subdirectories |

#### Example command
```
python tif_merge_convert.py .hgt --subdirs ../raw_data/srtm_raw
```
#### Summary
This script is not technical or critical enough to warrant an in-depth review of its mechanics, but in short:

This script locates every file within the specified target directory with the specified file extension when not in subdirs mode. When in subdirs mode, it locates every subdirectory of the target directory, and locates every file with the extension within those subdirectories. It then uses gdal to stitch the files together into a geotif (.tif), which is saved in the target directory.

### code_match.py
This is used to clip geotif raster data files to a specified AOI. 
#### Parameters
| Parameter | Usage | Function |
| --- | --- | --- |
| raster data paths | required comma-separated file paths (string) | paths of raster data files to clip |
| output data paths | required comma-separated fild paths (string) | paths to which clipped raster data files should be saved |
| bounding shape paths | required file path (string) | shapefile to be used to clip raster data |
| visualize data | (optional, default none) -v {all, none, comma-separated indices} | which clipped raster datasets to visualize |

#### Example command
The command below clips reduced_nlcd.tif to nlcd_clipped.tif and combined.tif to srtm_clipped.tif, using NEON_3D_Boundary as the clipping shape. It visualizes neither of the datasets.
```
python code_match.py ../raw_data/nlcd_raw/reduced_nlcd.tif,../raw_data/srtm_raw/combined.tif ../raw_data/nlcd_raw/nlcd_clipped.tif,../raw_data/srtm_raw/srtm_clipped.tif ../raw_data/neon_aoi/NEON_3D_Boundary.shp -v none -q 2
```
####  Summary
This script is not technical or critical enough to warrant an in-depth review of its mechanics, but in short:

This script parses the comma separated file paths from the command line arguments and iteratively loads all of the raster files. For each file, it clips the dataset to the specified shape using rioxarray.clip, and plots the resulting clipped raster data with matplotlib, saving the plot to figures/data.
If the index of the raster file in the comma separated list of files is one specified in the visualize data command line argument, the visualization will also be plotted with gui. Note that this can be a helpful sanity check, but may interrupt the execution of the code.

### analyze_clipped.py
This is used to gather information about raster data at a series of points. Within the scope of this project, it plots distributions of raster data values at GEDI centroids.

#### Parameters
| Parameter | Usage | Function |
| --- | --- | --- |
| raster data paths | required comma separated series of file paths (string) | raster data files to look at |
| points of interest shapefile | required file path (string) | shapefile containing points at which to analyze the raster data |
| test mode | (optional, default -1) -t int | whether to run on all points (-1) or run on the first n samples |
| verbosity | (optional, default 2) -q {0, 1, 2} | verbosity level, where 0 is less verbose and 2 is more verbose |

#### Example command
The command below will run basic analysis on srtm, nlcd, and ecostress WUE data at gedi centroids specified in GEDI_2B_clean.shp. This analysis will be run on all samples.
```
python analyze_clipped.py ../raw_data/srtm_raw/srtm_clipped.tif,../raw_data/nlcd_raw/nlcd_clipped.tif,../raw_data/ecos_wue/WUE_Median_Composite_AOI.tif ../raw_data/gedi_pts/GEDI_2B_clean.shp -t -1 -q 2
```

#### Summary 
This script is not technical or critical enough to warrant an in-depth review of its mechanics, but in short:

This script loads the shapefile (the GEDI data) and iteratively loads each raster data file. For each file, it iterates through every point (gedi centroid) and records the value of the raster data at that point. It then produces plots related to the distribution of the data at the points.
These plots are saved to figures/gedi_distributions.

### check_clip.py
This is used as an additional sanity check for the clipped raster data. Use this to visually inspect clipped raster data files.
#### Parameters
| Parameter | Usage | Function |
| --- | --- | --- |
| raster data paths | required comma separated series of file paths (string) | raster data files to visually inspect |

#### Example command
```
python check_clip.py ../raw_data/srtm_raw/srtm_clipped.tif ../raw_data/nlcd_raw/nlcd_clipped.tif
```

#### Summary
This script is not technical or critical enough to warrant an in-depth review of its mechanics; it simply loads and plots raster data files.


### reset_raster_nd.py
This is used to alter no-data values in raster data files.

#### Parameters
| Parameter | Usage | Function |
| --- | --- | --- |
| file path | required file path (string) | raster dataset to inspect and/or alter |
| nodata value | required nodata value (int) | value to set as no-data value, if in edit mode |
| edit mode | required {t, f} | whether to edit the no-data value or just inspect it (get it and print it out) |

#### Example command
```
python reset_rasted_nd.py ../raw_data/srtm_raw/srtm_clipped.tif 0 f
```

#### Summary 
This script is not technical or critical enough to warrant an in-depth review of its mechanics, but in short:

The SRTM data as I worked with it had no-data values that were not nan. This would in all likelihood not create problems elsewhere in the data creation pipeline, but it was inconsistent and created a visually unappealing background on plots of the raster datset. When not running in edit mode, this script loads the specified raster data file and obtains the original no-data value using rioxarray, then prints it. In edit mode, it does the same but then sets the no-data value to the one specified in the command line argument and saves the file.

### create_2.py
This is the central piece of the entire data creation pipeline, so I will outline this file in greater detail.
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
python create_2.py ../raw_data/srtm_raw/srtm_clipped.tif,../raw_data/nlcd_raw/nlcd_clipped.tif,../raw_data/slope_raw/slope_clipped.tif,../raw_data/aspct_raw/aspct_clipped.tif ../raw_data/gedi_pts/GEDI_2B_clean.shp ../raw_data/ecos_wue/wue_median_composite_clipped.tif 70 5 true ../data/data_h5test --lomem --gencoords --genetc --override --prescreen --h5mode=h5 --cfields=cover,pavd,fhd --orient=hwc --pad=1 --hashpad=10 --chunk=10  --q=2
```

Remaking (revising) a pre-existing dataset:
```
python create_2.py ../raw_data/srtm_raw/srtm_clipped.tif,../raw_data/nlcd_raw/nlcd_clipped.tif,../raw_data/slope_raw/slope_clipped.tif,../raw_data/aspct_raw/aspct_clipped.tif ../raw_data/gedi_pts/GEDI_2B_clean.shp ../raw_data/ecos_wue/wue_median_composite_clipped.tif 70 5 true ../data/data_h5test --lomem --gencoords --genetc --prescreen --h5mode=h5 --cfields=cover,pavd,fhd --orient=hwc --pad=1 --hashpad=10 --chunk=10  --q=2
```

#### Under the hood
##### Loading Raster Data
The file separated file paths are parsed and added to the list x_raster_locs. The raster files are loading with GDAL and information about nodata values and coordinate reference systems are recorded. The raster is then saved as a numpy array for ease of use. The below process is mirrored for loading the y raster
```python
for loc in x_raster_locs:
    tdataname = loc.split("/")[-1]                                        ### parsing file name
    xraster.append(gdal.Open(loc))                                        ### load raster file (geotif) with gdal
    layernames.append(tdataname.split(".")[0])                            ### parse layer name
    tdata_rband = xraster[-1].GetRasterBand(1)                            ### get nodata value
    ndv_vals.append(tdata_rband.GetNoDataValue())                         
    xr_rsize.append((xraster[-1].RasterXSize, xraster[-1].RasterYSize))   ### record dimensions of raster
    tulh, tpxh, _, tulv, _, tpxv = xraster[-1].GetGeoTransform()          ### get coordinate reference system parameters
    tpxv = abs(tpxv)                                                      ### account for sign change in array indexing and geographical coordinates
    xr_params.append((tulh, tulv, tpxh, tpxv))                            ### record crs parameters
    xr_npar.append(xraster[-1].ReadAsArray().transpose())                 ### record raster data as array
```
##### GEDI shapefile records
Part 1: Reading the shapefile and store the records in the list 'grecs'. Here pt is referencing the gedi data file path.
```python
xpoints.append(shapefile.Reader(pt))
grecs.append(xpoints[-1].shapeRecords())
```
Part 2: Building and saving metadata for the gedi data. We want to use every field that contains a keyword given in the -c command line argument. The code below iteratively locates these fields and saves their names for future reference and interpretability. The order of fields in this list also dictates the order of gedi channels when the unified datacubes are constructed. The code also builds a point indexer (ptindexer) that is saved for future use. This allows for easy translation between the keyword, the index of the field in the list of fields containing a keyword, and the index of the field in the shapefile. It also allows for separation of shapefile data in the event that another shapefile point-data source should be incorporated into the dataset.
Here critical_fields is a list of keywords.
```python
for i in range(len(critical_fields)):
    ptl_idx = 0
    for j in range(len(xpoints[-1].fields)):
        if critical_fields[i] in xpoints[-1].fields[j][0]:
            ### given the layer (key) provides the ptfile #, critical field #, field index within that shape,
            ### and id within the np array (if exists)
            ptindexer[len(ptlayers)] = (len(xpoints) -1, i, j, ptl_idx)
            ptlayers.append(critical_fields[i] + "_" + str(ptl_idx))
            ptlnames.append(xpoints[-1].fields[j][0])
            ptl_idx += 1
```
Part 3: Reformat data if low-memory mode is on. This uninteresting section of the code loads GEDI data from the fields selected in the previous step and saves them to a much less cumbersome csv file.
Part 4: Clear GEDI data from memory and reload them from the csv files we just made, if low memory mode is on.
##### Indexing helpers
With all of the different datasets, coordinate systems, and indexing systems involved, the following code helps streamline some ways in which we will be interacting with the data a lot later.
cgetter returns the coordinates in the GEDI data's crs of a point at a specified index:
```python
def cgetter(index, xy):
    if lo_mem:
        return npcoords[index, xy]
    else:
        return grecs[0][index].record[3-xy]
```
clen returns the number of centroids in the GEDI dataset:
```python
def clen():
    if lo_mem:
        return npcoords.shape[0]
    else:
        return len(grecs[0])
```
pgetter returns the value of a specified field (layer) for a specified point (index):
```python
def pgetter(layer, index):
    #print(layer)
    if lo_mem:
        ### use crit_npar
        return crit_npar[ptindexer[layer][1]][index, ptindexer[layer][3]]
    else:
        return grecs[ptindexer[layer][0]][index].record[ptindexer[layer][2]]
```
##### CRS helpers
The following are critical functions. The first converts a coordinate in a crs to the encompassing index in the raster array. Here, cx and cy are the crs coordinates whereas ix and iy are array indices, and the other parameters specify the crs.
```python
def coords_idx(cx, cy, ulh, ulv, psh, psv):
    ix = int((cx - ulh)/psh)
    iy = int((ulv - cy)/psv)
    return ix, iy
```
The second converts an array index to a coordinate in a crs. This is tricky because the crs coordinates are continuous but the pixels represented by array values are some meters wide and high. By default, it will compute the coordinate of the upper left corner of the pixel, but if the mode is set to center (mode='ctr') then it will compute the coordinate of the center of the pixel.
```python
def idx_pixctr(ix, iy, ulh, ulv, psh, psv, mode='ul'):
    offsetx = 0
    offsety = 0
    if mode=='ctr':
        offsetx = psh/2
        offsety = psv/2
    cx = ulh + (ix * psh) + offsetx
    cy = ulv - (iy * psv) + offsety
    return cx, cy
```
##### Nearest neighbor interpolation method
The problem with applying nearest neighbor interpolation to this problem is that there are hundreds of thousands of GEDI centerpoints and millions of grid squares in the ECOSTRESS AOI raster - and even worse, the interpolation needs to be done on a 16*16 grid within each GEDI grid square. The number of options involved makes naive methods of finding nearest neighbors too computationally difficult to be practical here. This algorithm takes advantage of some structural properties of the GEDI centroids: they are somewhat evenly spaced across the AOI, and there are a small number of other centroids within a short distance like 70m of each centroid.

First, it hashes the indices of each centroid to a large 70m array mirroring the ecostress raster array. The snippet below creates a large numpy array and fills it with empty lists. hash_pad is provided in the command line arguments, and creates a buffer around the edge of the hash array so that the algorithm does not look in an out-of-bounds array location in following steps.
```python
ygrid_pt_hash = np.zeros((yrsize[0] + (2*hash_pad), yrsize[1] + (2*hash_pad)), dtype='object')
for i in range(ygrid_pt_hash.shape[0]):
    for j in range(ygrid_pt_hash.shape[1]):
        ygrid_pt_hash[i, j] = []
```
Now the hashing can take place. for each GEDI centroid, its in-crs coordinates are mapped to array indices. If the sample falls out of bounds, some diagnostic information is printed, since this will create problems down the road. If not, the index is added to the list in the appropriate cell of the large array:
```python
for i in range(clen()):
    xi, yi = coords_idx(cgetter(i, 0), cgetter(i, 1), yulh, yulv, ypxh, ypxv)
    if xi+hash_pad < 0 or yi+hash_pad < 0 or xi > yrsize[0]+(hash_pad*2) or yi > yrsize[1]+(2*hash_pad):
        print("big uh-oh!!!")
        print(i)
        print(cgetter(i, 0), cgetter(i, 1))
        print(xi, yi)
    else:
        actual_added += 1
        ygrid_pt_hash[xi+hash_pad, yi+hash_pad].append(i)
```
The final step is using this hash to compute a subset of gedi centroids that nearest neighbors can be computed from for each GEDI grid square. The algorithm starts at the grid square in question and works outwards in rings of grid squares, building a list of all gedi centroids in each visited grid square.

The smallest possible distance between a centroid encountered in ring n and the nearest point in the central grid square is n-1. The largest possible distance between a centroid encountered in ring n and the farthest possible point in the central grid square is sqrt(2) * n. Because of this, if the first centroid c1 is encountered in ring n, There may still be centroids closer than c1 to some point within the central grid square until ring sqrt(2)*n + 1. This is made to be an integer with ceiling(sqrt(2)*n) + 1.

In short, if the first centroid is found in ring n, then the furthest possible centroid that could still be the nearest neighbor to some point in the grid square is in ring ceiling(sqrt(2)*n) + 1. So the algorithm can stop at that point.
```python
def krings(x_in, y_in, min_k):
    ring_size = 0
    found_list = []
    cap = -1
    ### continue to look at the next ring until the stopping point has been reached
    while(cap < 0 or ring_size <= cap):
        i_boundaries = [max(0-hash_pad, x_in-ring_size), min(yrsize[0]+hash_pad, x_in+ring_size+1)]
        j_boundaries = [max(0-hash_pad, y_in-ring_size), min(yrsize[1]+hash_pad, y_in+ring_size+1)]
        ### iterate throigh all the points in the curent ring
        for i in range(i_boundaries[0], i_boundaries[1]):
            for j in range(j_boundaries[0], j_boundaries[1]):
                if i == i_boundaries[0] or i+1 == i_boundaries[1] or j == j_boundaries[0] or j+1 == j_boundaries[1]:
                    ### if any centroids have been hashed to this location, copy them over to our short list.
                    ### if this is the first centroid to be found (cap = -1) then the stopping point can be set.
                    if len(ygrid_pt_hash[i+1, j+1]) > 0:
                        if cap == -1:
                            cap = max(math.ceil(mem_root2 * ring_size), 1) + 1
                        for k in ygrid_pt_hash[i+1, j+1]:
                            found_list.append(k)
        ring_size += 1
    return found_list, ring_size
```

##### Building the New Dataset
Some setup is required for building the new dataset.
The database used to store y values and some metadata is initialized below, as is the h5 dataset and a numpy array used to store an h5 chunk before it is written to the h5 dataset (which greatly speeds up the process, based on experimentation).
If shuffle_order is set to true in the command line arguments, the order in which the raster indices are iterated through is shuffled:
```python
irange_default = np.arange(yrsize[0])
jrange_default = np.arange(yrsize[1])
np.random.shuffle(irange_default)
np.random.shuffle(jrange_default)
```

Finally, the raster squares can be iterated through to build the dataset. First, the code below checks whether the y raster square is the no-data value, which would indicate that the square is outside of the AOI. Then, it initializes a datacube, which is used for building the .csv-style dataset.
```python
for i in irange_default:
    for j in jrange_default:
        if y_npar[i, j] != yndv:
            if not h5_mode or (h5_mode and h5_scsv):
                if channel_first:
                    x_img = np.zeros((channels, imgsize+(2*pad_img), imgsize+(2*pad_img)))
                else:
                    x_img = np.zeros((imgsize+(2*pad_img), imgsize+(2*pad_img), channels))
            nlcd_count = 0
```

Still within the no-data conditional, the code below iterates through each raster dataset and through each 5m pixel within the 70m ecostress grid square (plus padding pixels) to resample raster values at the center of each pixel. The resampling is performed by computing the i and j index offsets, or the location of the centerpoint of the pixel if the raster grid square were 1x1. This is the location that should be resampled in the ecostress raster array index coordinate system. This is then converted to x and y coordinates in the ecostress raster geospatial coordinate system with idx_pixtr(), described above. This crs is shared with the x raster data, so the coordinates are converted again into the x raster array index coordinate system with coords_idx(), again described above. The value at these indices are the values that we want to sample.
The last section of the code stores the data in the sample based on h5 mode and sample orientation.
```python
            for k in range(len(xr_npar)):
                for si in range(0 - pad_img, imgsize+pad_img):
                    for sj in range(0 - pad_img, imgsize+pad_img):
                        sxoffset = ((2 * si) + 1) / (2 * imgsize)
                        syoffset = ((2 * sj) + 1) / (2 * imgsize)
                        tempx, tempy = idx_pixctr(i + sxoffset, j + syoffset, yulh, yulv, ypxh,
                                ypxv, mode = 'ul')
                        tempi, tempj = coords_idx(tempx, tempy, xr_params[k][0], xr_params[k][1],
                                xr_params[k][2], xr_params[k][3])
                        ### save to datacube one way or another
                        if not h5_mode or (h5_mode and h5_scsv):
                            if channel_first:
                                x_img[k, si+pad_img, sj+pad_img] = xr_npar[k][tempi, tempj]
                            else:
                                x_img[si+pad_img, sj+pad_img, k] = xr_npar[k][tempi, tempj]
                        if h5_mode:
                            if channel_first:
                                h5_chunk[h5tid, k, si+pad_img, sj+pad_img] = xr_npar[k][tempi, tempj]
                            else:
                                h5_chunk[h5tid, si+pad_img, sj+pad_img, k] = xr_npar[k][tempi, tempj]
```

Also within the no-data conditional, the code below uses krings() described above to retreive a list of possible nearest neighbors for the ecostress grid square at this index. Next, it iterates through each of the 5m pixels. It computes the i and j offsets again, and converts the indices back to the geospatial crs.
```python
            k_ids, rings = krings(i, j, k_approx)
            for si in range(0-pad_img, imgsize+pad_img):
                for sj in range(0-pad_img, imgsize+pad_img):
                    sxoffset = ((2 * si) + 1) / (2 * imgsize)
                    syoffset = ((2 * sj) + 1) / (2 * imgsize)
                    tempx, tempy = idx_pixctr(i + sxoffset, j + syoffset, yulh, yulv, ypxh,
                            ypxv, mode='ul')
```

It then finds the closest gedi centroid to the center of the pixel with brute force, which is ok because the number of centroids involved is very small relative to the total number in the dataset. The index of the centroid with the smallest distance is saved as minpt, and its distance is saved as mindist. The data from each relevant field from the closest gedi centroid is then recorded, based on h5 mode and sample orientation. Finally, mindist and minpt are recorded - This is only shown for one case due to its repetitive nature.
```python
                    mindist = 100000
                    minpt = None
                    for pt_idx in k_ids:
                        tdist = cdist(npcoords[pt_idx, 0], npcoords[pt_idx, 1], tempx, tempy)
                        if tdist < mindist:
                            mindist = tdist
                            minpt = pt_idx
                    for m in range(len(ptlayers)):
                        if not h5_mode or (h5_mode and h5_scsv):
                            if channel_first:
                                x_img[len(xr_npar) + m, si+pad_img, sj+pad_img] = pgetter(m, minpt)
                            else:
                                x_img[si+pad_img, sj+pad_img, len(xr_npar) + m] = pgetter(m, minpt)
                        if h5_mode:
                            if channel_first:
                                h5_chunk[h5tid, len(xr_npar)+m, si+pad_img, sj+pad_img] = pgetter(m, minpt)
                            else:
                                h5_chunk[h5tid, si+pad_img, sj+pad_img, len(xr_npar)+m] = pgetter(m, minpt)
                    if not h5_mode or (h5_mode and h5_scsv):
                        if channel_first:
                            x_img[len(xr_npar) + len(ptlayers), si+pad_img, sj+pad_img] = minpt
                            x_img[len(xr_npar) + len(ptlayers) + 1, si+pad_img, sj+pad_img] = mindist
```

Not all ecostress raster grid squares have a gedi centroid within or near them. These samples might be unreliable if gedi centroid data is interpolated from centroids that are many meters outside of the grid square. However, setting this threshold in the dataset building process will make it difficult and time consuming to check later. Therefore, the distance from the center of the grid square to the nearest centroid is recorded as metadata, so that thresholding can take place later with greater flexibility. There are an even number of pixels, so no one pixel is at the center, so the code below computes the average distance from the four pixels surrounding the center. Only one case is shown due to the repetitive nature of this task:
```python
            if h5_mode:
                if channel_first:
                    avg_mid_dist = h5_chunk[h5tid, -1, (imgsize + (pad_img * 2)) // 2, (imgsize + (pad_img * 2)) // 2] / 4
                    avg_mid_dist += h5_chunk[h5tid, -1, (imgsize + (pad_img * 2) - 1) // 2, (imgsize + (pad_img * 2)) // 2] / 4
                    avg_mid_dist += h5_chunk[h5tid, -1, (imgsize + (pad_img * 2)) // 2, (imgsize + (pad_img * 2) - 1) // 2] / 4
                    avg_mid_dist += h5_chunk[h5tid, -1, (imgsize + (pad_img * 2) - 1) // 2, (imgsize + (pad_img * 2) - 1) // 2] / 4
```

Finally, the datacube created above is recorded based on h5mode, and administrative work related to h5 chunks is performed in that case.
```python
            if ...
                if not skip_save and (not h5_mode or (h5_mode and h5_scsv)):
                    if channel_first:
                        np.savetxt(fs_loc + "/datasrc/x_img/x_" +str(nsuccess)+ ".csv", x_img.reshape(x_img.shape[0], -1),
                                delimiter=",", newline="\n")
                    else:
                        np.savetxt(fs_loc + "/datasrc/x_img/x_" +str(nsuccess)+ ".csv", x_img.reshape(-1, x_img.shape[2]),
                                delimiter=",", newline="\n")
                if not skip_save and h5_mode:
                    h5tid += 1
                    h5len += 1
                    if h5tid == h5chunksize:
                        h5chunkid += 1
                        h5dset.resize(h5len, axis=0)
                        h5dset[h5len-h5chunksize:h5len,:,:,:] = np.array(h5_chunk[:,:,:,:])
                        h5_chunk = np.zeros(h5_chunk.shape)
                        h5_chunk.fill(-1)
                        h5tid = 0
                if not skip_save:
                    database.append(["/datasrc/x_img/x_" + str(nsuccess) + ".csv", y_npar[i, j], nsuccess, i, j, avg_mid_dist])
                nsuccess += 1
```
The X data is either saved to a csv above or automatically saved to a hdf5 database. The y data is saved to a csv as well, concluding the dataset building process.

### build_train_val_test.py
This script is used to partition the data into training and test data, and create crossvalidation folds. In reality, these splits are done by building a list of indices and subdividing the array with scikit-learn's train_test_split
#### Parameters
| parameter | Usage | Function |
| --- | --- | --- |
| dataset | required string | name of dataset to partition | 
| split_name | required string | name if the split to create (since each dataset can have more than one split)|
| folds | required int | number of cross-validation folds to create |
| test fraction | required number in \[0, 1\] | fraction of total data to be set aside for the test set |
| validation fraction | required float in \[0, 1\] | fraction of data for each validation fold to be given to validation set as opposed to train set |

#### Recommended commands
Create 5 validation folds with 0.2 and 0.3 splits:
```
python build_train_val_test.py dataset_name split_name 5 0.2 0.3
```
#### Under the hood
The code manages directories for this train/validation/test split, creating a directory for the split (if one does not already exist) and then creates a directory for each validation fold, then divides up the data. It reads in the y data and metadata csv and determines how many samples are included. It then creates an index array which is randomly divided into a test set and a remaining set for training and validation with scikit-learn's model_selection.train_test_split. This produces a test set with test_fraction of the data and a train/validation set with 1-test_fraction of the data:
```python
rawdata = pd.read_csv(prefix + "ydata.csv").to_numpy()
idx_split = np.arange(rawdata.shape[0])
train_ids, test_ids, = train_test_split(idx_split, test_size=test_frac)
```
for each validation fold, the remaining train/validation set is divided randomly into a validation set with validation_frac and a train set with 1-validation_frac of the train/validation data. This is done with replacement, so each split has the same total number of samples as the train/validation set:
```python
for i in range(folds):
    print("building validation fold " + str(i))
    train_i, val_i = train_test_split(train_ids, test_size=val_frac)
    print("fold ", len(train_i), len(val_i))
```
Each of these index arrays are saved as csv files to their respective fold directories, concluding the contents of this file.

### h5_sanitycheck.py
Verify that the samples created in csv mode are the same (up to some small epsilon) as the ones created in h5 mode.
This script has no parameters. This will only work if the dataset is generated in h5_both mode (or if random shuffle is off), so that the indices of the samples in the raster and the meta csv are consistent between the two methods.
