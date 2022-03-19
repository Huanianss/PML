import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm, laplace

x = np.linspace(-4, 4, 100)

gauss = norm.pdf(x)
y1 = t.pdf(x, 1, 0, 1)
y2 = t.pdf(x, 2, 0, 1)
l = laplace.pdf(x, 0, 1 / np.sqrt(2))

plt.plot(x, gauss, 'k--', label='Gaussian')
plt.plot(x, y1, 'b--', label='Student 1')
plt.plot(x, y2, 'g--', label='Student 1')
plt.plot(x, l, 'r', label='Laplace')
plt.legend()
plt.show()

plt.semilogy(x, gauss, 'k--', label='Gaussian')
plt.semilogy(x, y1, 'b--', label='Student 1')
plt.semilogy(x, y2, 'g--', label='Student 1')
plt.semilogy(x, l, 'r', label='Laplace')
plt.legend()
plt.show()
