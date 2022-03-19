import numpy as np
import matplotlib.pyplot as plt


y = np.linspace(0.01, 1, 100)
plt.plot(y, 1 / 2 / np.sqrt(y))
plt.axis([0, 1, 0, 6])
plt.show()

N = 1000
x = np.random.rand(N)
y = x ** 2
plt.hist(y, 20, density=True, weights=[1 / N] * N)
plt.show()
