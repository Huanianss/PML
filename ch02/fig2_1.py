import numpy as np
import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.bar(np.arange(1, 5, 1), 0.25, 0.7)
plt.axis([0, 5, 0, 1])

plt.subplot(1, 2, 2)
plt.bar(1, 1, 0.7)
plt.axis([0, 5, 0, 1.1])
plt.show()
