import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, laplace, norm

np.random.seed(0)
a = np.random.randn(30)
outliers = np.array([8, 8.75, 9.5])

print(a)
print(outliers)

x = np.linspace(-5, 10, 300)
norm1 = norm.fit(a)
norm1 = norm.pdf(x, norm1[0], norm1[1])

t1 = t.fit(a)
t1 = t.pdf(x, t1[0], t1[1], t1[2])

l1 = laplace.fit(a)
l1 = laplace.pdf(x, l1[0], l1[1])

plt.hist(a, 10, weights=[1 / 30] * 30, rwidth=0.8, edgecolor='k')
plt.plot(x, norm1, 'k--', label='Gaussian')
plt.plot(x, t1, 'r', label='Student t')
plt.plot(x, l1, 'b--', label='Laplace')
plt.axis([-5, 10, 0, 0.6])
plt.legend()
plt.show()

a = np.concatenate([a, outliers])
norm1 = norm.fit(a)
norm1 = norm.pdf(x, norm1[0], norm1[1])

t1 = t.fit(a)
t1 = t.pdf(x, t1[0], t1[1], t1[2])

l1 = laplace.fit(a)
l1 = laplace.pdf(x, l1[0], l1[1])

plt.hist(a, 10, weights=[1 / 33] * 33, rwidth=0.8, edgecolor='k')
plt.plot(x, norm1, 'k--', label='Gaussian')
plt.plot(x, t1, 'r', label='Student t')
plt.plot(x, l1, 'b--', label='Laplace')
plt.axis([-5, 10, 0, 0.6])
plt.legend()
plt.show()
