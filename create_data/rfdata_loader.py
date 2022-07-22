import numpy as np

def rfloader(fpath):
    with open(fpath) as f:
        alllines = f.readlines()
    #print(alllines)
    #print(len(alllines))
    line = 0
    rawdata = []
    fnames = []
    for fline in alllines:
        #print(fline[:-1])
        if line == 0:
            fnames = list(fline[:-1].split(","))
        else:
            try:
                rawdata.append(list([float(ii) for ii in fline[:-1].split(",")]))
            except:
                print(fpath)
                print(line)
                print(fline)
        line += 1

    return np.array(rawdata), fnames

def piloader(fpath):
    # load special dict format... pt indexer file
    return_dict = {}
    with open(fpath) as f:
        alllines = f.readlines()
    for fline in alllines:
        key, arraycon = fline.split(":")
        return_dict[int(key)] = [int(ii) for ii in arraycon[:-1].split(",")]

    print(return_dict)
    return return_dict

def d1loader(fpath):
    #load 1d csv
    return_list = []
    with open(fpath) as f:
        alllines = f.readlines()
    for fline in alllines:
        if fline != "":
            return_list.append(fline.split(",")[:-1])
    return return_list

### Test
#piloader("../data/data_interpolated/point_reformat/test_data.txt")
