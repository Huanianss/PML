import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

rv = norm(0, 2)
X = np.linspace(-3, 3, 500)

plt.plot(X, rv.cdf(X))
plt.title("Gaussian cdf")
plt.show()

# 3 sigma
sigma1 = 2 * rv.cdf(1) - 1
sigma2 = 2 * rv.cdf(2) - 1
sigma3 = 2 * rv.cdf(3) - 1
print('1 sigma:', sigma1)
print('2 sigma:', sigma2)
print('3 sigma:', sigma3)


plt.plot(X, rv.pdf(X))
plt.fill_between(np.linspace(-1, 1), rv.pdf(np.linspace(-1, 1)), alpha=0.3)
plt.fill_between(np.linspace(-2, 2), rv.pdf(np.linspace(-2, 2)), alpha=0.3)
plt.fill_between(np.linspace(-3, 3), rv.pdf(np.linspace(-3, 3)), alpha=0.3)

plt.title("Gaussian pdf")
plt.show()
