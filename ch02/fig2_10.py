import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(-4, 4, 50)
sigmoid_a = 1 / (1 + np.exp(-a))
h = (a > 0)
print(h)
plt.subplot(1, 2, 1)
plt.plot(a, sigmoid_a)
plt.subplot(1, 2, 2)
plt.plot(a, h)
plt.show()
