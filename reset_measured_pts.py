import numpy as np

x = np.load("measured_points/T1_A1.npy")
print(x.shape)
x = x.reshape((1, 3))
print(x.shape)
x = x[0, :]
print(x)
np.save("measured_points/T1_A1.npy", x)