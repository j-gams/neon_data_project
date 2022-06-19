### WHAT DOES THIS CODE DO?
### - finds all files with specified extension in specified directory
### - if in subdirs mode, looks for files in every subdirectory of specified directory instead
###   - subdirs mode useful when running on files that came in individual .zip archives, for example
###   - so the files don't have to be moved into one directory
### - converts them to geotif (.tif) files
### - if more than one file is detected in a directory, merges them into one .tif with gdal

### REQUIREMENTS
### - Packages
###     - gdal
### - Directory structure
###   - each set of files to convert needs to be in a separate directory
###     files may be unintentionally merged otherwise

### command line arguments:
### target extension    [ie ".tif", required]
### subdir mode         [--subdirs or blank (defaults to no subdirs if blank)]
### target directories  [list as many directories as needed]

### Usage Example:
### python tif_merge_convert.py .hgt --subdirs ../raw_data/srtm_raw
### above   .hgt is the target extension
###         --subdirs indicates that the program should run in subdirs mode
###         ../raw_dat/srtm_raw is the directory to look at

import sys

### handling command line arguments
ext = ""
dirs = []
do_subdirs = False
print(sys.argv)
if len(sys.argv) <= 2:
    print("please provide file extension to target (eg .hgt for srtm data)")
    print("please provide series of directories including files to convert")
    sys.exit("missing command line arguments!")
else:
    ext = sys.argv[1]
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == "--subdirs":
            do_subdirs = True
        else:
            dirs.append(sys.argv[i])
import os

### iterate through every directory passed to the program
if do_subdirs:
    print("running in subdirs mode")
for dirpath in dirs:
    ### used to aggregate file names so they can be merged
    ### and to tell if there are enough to merge
    idx = 0
    merge = ""

    ### run in subdirs mode -- look inside subdirectories of target directory
    if do_subdirs:
        ### store file paths
        dircollect = []
        ### find all subdirectories of target directory
        for odir in os.listdir(dirpath):
            if not os.path.isfile(dirpath + "/" + odir):
                dircontents = os.listdir(dirpath + "/" + odir)
                ### find all files within subdirectory with target extension. Keep track of their paths
                for item in dircontents:
                    if os.path.isfile(dirpath + "/" + odir + "/" + item):
                        dircollect.append((dirpath + "/" + odir + "/" + item))
        ### convert all files found above to geotif files
        ### save them in the parent directory (target directory, not subdirectory)
        for elt in dircollect:
            tname = elt.split("/")[-1].split(".")[0]
            os.system("gdal_translate -of GTiff " + elt + " " + dirpath + "/" + tname + ".tif")
            merge += dirpath + "/" + tname + ".tif "
            idx += 1
    ### run in default mode -- only look at files within target directory, ignore subdirectories
    else:
        dircontents = os.listdir(dirpath)
        ### find and convert files with the right extension
        for item in dircontents:
            if os.path.isfile(dirpath + "/" + item):
                if item.split(".")[-1] == ext:
                    # gotem
                    tname = item.split(".")[0]
                    os.system("gdal_translate -of GTiff " + dirpath + "/" + item + " " + dirpath + "/" + tname + ".tif")
                    merge += dirpath + "/" + tname + ".tif "
                    idx += 1
    ### if more than 1 file was found, merge them into a combined geotif file
    ### this file is named combined.tif in the target directory
    if idx > 1:
        print("more than one file detected for", dirpath)
        print("merging into one geotif")
        os.system("gdal_merge.py -o " + dirpath + "/combined.tif " + merge)
        print("done merging into", dirpath + "/combined.tif")
    else:
        print("only one file detected. No files to merge.")
print("done")

