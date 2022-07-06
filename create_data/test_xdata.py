import numpy as np
import matplotlib.pyplot as plt

x0 = 3
x1 = 14
x2 = 14
#320


load_arr = np.genfromtxt("../data/data_interpolated/datasrc/x_img/x_114.csv", delimiter=',')
loaded_reshaped = load_arr.reshape(-1, 14, 14)
plt.imshow(loaded_reshaped[0])
plt.show()
plt.cla()
plt.imshow(loaded_reshaped[1])
plt.show()
plt.cla()
plt.imshow(loaded_reshaped[-1])
plt.show()
plt.cla()

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