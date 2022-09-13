### Written by Jerry Gammie @j-gams

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

### :set number

dataset = sys.argv[1]
mini_name = sys.argv[2]
force = True
folds = int(sys.argv[3])
test_frac = float(sys.argv[4])#0.2
val_frac = float(sys.argv[5])#0.3
if len(sys.argv) > 6:
    print("obtaining np random seed")
    npseed = int(sys.argv[6])
    np.random.state(npseed)
    print("numpy random state set to", npseed)
if not os.path.isdir(dataset + "/fold_data"):
    os.mkdir(dataset + "/fold_data")
if not os.path.isdir(dataset + "/fold_data/" + mini_name):
    os.mkdir(dataset + "/fold_data/" + mini_name)
    os.mkdir(dataset + "/fold_data/" + mini_name + "/test")
    for i in range(folds):
        os.mkdir(dataset + "/fold_data/" + mini_name + "/train_fold_" + str(i))
elif not force:
    sys.exit("data split already exists")

print("building test set")
prefix = dataset + "/datasrc/"
rawdata = pd.read_csv(prefix + "ydata.csv").to_numpy()
idx_split = np.arange(rawdata.shape[0])
print("ntotal", rawdata.shape[0])
train_ids, test_ids, = train_test_split(idx_split, test_size=test_frac)
print("test ", len(test_ids))
np.savetxt(dataset + "/fold_data/" + mini_name + "/test/test_set.csv", test_ids, delimiter=",")
for i in range(folds):
    print("building validation fold " + str(i))
    train_i, val_i = train_test_split(train_ids, test_size=val_frac)
    print("fold ", len(train_i), len(val_i))
    np.savetxt(dataset+"/fold_data/" + mini_name + "/train_fold_" + str(i) + "/val_fold.csv", val_i, delimiter=",")
    np.savetxt(dataset+"/fold_data/" + mini_name + "/train_fold_" + str(i) + "/train_fold.csv", train_i, delimiter=",")

### TODO - save meta file
### 
os.system("rm " + dataset + "/fold_data/" + mini_name + "/meta.txt")
os.system("touch " + dataset + "/fold_data/" + mini_name + "/meta.txt")
os.system('echo "folds: ' + str(folds) + '" >> ' + dataset + '/fold_data/' + mini_name + '/meta.txt')
print("done")
#print(rawdata)
