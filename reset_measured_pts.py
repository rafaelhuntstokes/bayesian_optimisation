import numpy as np

x = np.load("measured_points/T2_theta2.npy")
print(x.shape)
print(x)
x = x[:-1, :]
print(x)
np.save("measured_points/T2_theta2.npy", x)