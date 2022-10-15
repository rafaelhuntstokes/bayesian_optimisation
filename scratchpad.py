import numpy as np

# x0_res= 100
# x1_res = 100
# x2_res = 50
# x0_range = [0,5]
# x1_range = [0,10]
# x2_range = [0,20]

# pred_points = np.zeros((x0_res, x1_res, x2_res, 3))

# for (i, x0) in enumerate(np.linspace(x0_range[0], x0_range[1], x0_res)):
#     for (j, x1) in enumerate(np.linspace(x1_range[0], x1_range[1], x1_res)):
#         for (k, x2) in enumerate(np.linspace(x2_range[0], x2_range[1], x2_res)):
#             pred_points[i,j,k, 0] = x0
#             pred_points[i,j,k, 1] = x1
#             pred_points[i,j,k, 2] = x2

# pred_points = pred_points.reshape((x0_res * x1_res * x2_res, 3))
# print(pred_points.shape)

a = np.array([[1,2], [3,4], [5,6]])
b = np.array([[2,2], [2,2], [2,2], [2,2]])


# print(a.shape)
print(a)
# a = a[:,None,:]
# b = b[None,:,:]
j1 = a[:,None,:]
j2 = b[None,:,:]
c = a[None, :, :]
d = b[:, :, None]

print(c.shape)
print(d.shape)
# result = np.linalg.norm(a- b, axis = 2) 
result1 = j1 - j2
result2 = c - d 

print(result1 == result2)
# print(result)
# print(result.shape)
# # print(result)
# for i in range(result.shape[0]):
#     for j in range(result.shape[1]):
#         for k in range(result.shape[2]):
#             print(f"{(i,j,k)}: {result[i,j,k]}")
