import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

X = np.arange(0, 11, 1)
bi = binom(10, 0.25)

y = bi.pmf(X)

plt.bar(X, y)
plt.title(r'$\theta =0.25$')
plt.show()

bi = binom(10, 0.9)
y = bi.pmf(X)

plt.bar(X, y)
plt.title(r'$\theta =0.9$')
plt.show()