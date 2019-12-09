import numpy as np

arr = np.empty((1,3))
arr = np.vstack((arr, np.array([1,2,3])))
print(arr)