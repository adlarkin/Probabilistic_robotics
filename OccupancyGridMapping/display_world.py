import numpy as np
import matplotlib.pyplot as plt

world = np.load("./final_world.npy", allow_pickle=True)
plt.imshow(world, cmap='gray')
plt.show()