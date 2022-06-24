import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = sys.argv[1]
mini_name = sys.argv[2]
force = True
folds = 1
test_frac = 0.2
val_frac = 0.3
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
train_ids, test_ids, = train_test_split(idx_split, test_size=test_frac)
np.savetxt(dataset + "fold_data/" + mini_name + "/test/test_set.csv", test_ids, delimiter=",")
for i in range(folds):
    print("building validation fold " + str(i))
    train_i, val_i = train_test_split(train_ids, test_size=val_frac)
    np.savetxt(dataset+"fold_data/" + mini_name + "/train_fold_" + str(i) + "/val_fold.csv", val_i, delimiter=",")
    np.savetxt(dataset+"fold_data/" + mini_name + "/train_fold_" + str(i) + "/train_fold.csv", train_i, delimiter=",")

print("done")
#print(rawdata)
