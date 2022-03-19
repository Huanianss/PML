import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

s = 10000

x = np.random.beta(1, 5, s)

plt.hist(x, 40, density=True, rwidth=0.7)
plt.show()

x = np.random.beta(1, 5, 5 * s)
x = x.reshape(5, 10000)
x = np.mean(x, 0)
plt.hist(x, 40, density=True, rwidth=0.7)
plt.show()
