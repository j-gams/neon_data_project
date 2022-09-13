### Written by Jerry Gammie @j-gams

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
x0 = 3
x1 = 14
x2 = 14
#320

load_in = int(sys.argv[1])
h5_src = h5py.File('../data/data_h51/datasrc/x_h5.h5','r')
h5_dataset = h5_src["data"]
print("h5 shape:", h5_dataset.shape)


#for i in range(100000):
if True:
    i = load_in
    loaded_reshaped = np.array(h5_dataset[i])
    if len(np.unique(loaded_reshaped[:,:,1])) > 3:
        plt.imshow(loaded_reshaped[:,:,0])
        plt.title("SRTM")
        plt.colorbar()
        plt.savefig("../figures/samples/srtm.png")
        plt.show()
        plt.cla()
        plt.imshow(loaded_reshaped[:,:,1])
        plt.title("NLCD")
        plt.colorbar()
        plt.savefig("../figures/samples/nlcd.png")
        plt.show()
        plt.cla()
        plt.imshow(loaded_reshaped[:,:,2])
        plt.title("Slope")
        plt.colorbar()
        plt.savefig("../figures/samples/slope.png")
        plt.show()
        print(i)
        plt.imshow(loaded_reshaped[:, :, 3])
        plt.title("Aspect")
        plt.colorbar()
        plt.savefig("../figures/samples/aspect.png")
        plt.show()

"""    if len(np.unique(loaded_reshaped[:,:,-2])) > 3:
        plt.imshow(loaded_reshaped[:,:,0])
        plt.savefig("../figures/samples/srtm.png")
        plt.show()
        plt.cla()
        plt.imshow(loaded_reshaped[:,:,1])
        plt.savefig("../figures/samples/nlcd.png")
        plt.show()
        plt.cla()
        plt.imshow(loaded_reshaped[:,:,-2])
        plt.savefig("../figures/samples/nearest.png")
        plt.show()
        print(i)"""

"""load_arr = np.genfromtxt("../data/data_interpolated/datasrc/x_img/x_114.csv", delimiter=',')
loaded_reshaped = load_arr.reshape(-1, 14, 14)
plt.imshow(loaded_reshaped[0])
plt.show()
plt.cla()
plt.imshow(loaded_reshaped[1])
plt.show()
plt.cla()
plt.imshow(loaded_reshaped[-1])
plt.show()
plt.cla()"""

"""
#for i in range(1000):
#    load_arr = np.genfromtxt("../data/data_interpolated/datasrc/x_img/x_"+str(i)+".csv", delimiter=',')
#    #print(load_arr)
#
#    loaded_reshaped = load_arr.reshape(-1, 14, 14)
#    if len(np.unique(loaded_reshaped[2])) >= 3:
#        print(i)
#        plt.imshow(loaded_reshaped[2])
#        plt.show()
#        plt.cla()"""
#        """ plt.imshow(loaded_reshaped[1])
#        plt.show()
#        plt.cla()
#        plt.imshow(loaded_reshaped[0])
#        plt.show()
#        plt.cla()"""