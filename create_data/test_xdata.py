import numpy as np
import matplotlib.pyplot as plt

x0 = 3
x1 = 14
x2 = 14
#320

for i in range(1000):
    load_arr = np.genfromtxt("data_nn_gedi/datasrc/x_img/x_"+str(i)+".csv", delimiter=',')
    #print(load_arr)

    if len(np.unique(load_arr[2])) > 2:
        loaded_reshaped = load_arr.reshape(3, 14, 14)
        plt.imshow(loaded_reshaped[2])
        plt.show()
        plt.cla()
        plt.imshow(loaded_reshaped[1])
        plt.show()
        plt.cla()
        plt.imshow(loaded_reshaped[0])
        plt.show()
        plt.cla()