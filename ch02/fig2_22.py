import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(1, 7, 1000)
y = np.random.randint(1, 7, 1000)
plt.figure()
plt.hist(x + y, 11, rwidth=0.5, weights=[1 / 1000] * 1000)
plt.grid(axis='y')
plt.show()
