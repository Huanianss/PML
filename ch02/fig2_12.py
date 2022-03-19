import numpy as np
import matplotlib.pyplot as plt


def softmax(a, T=1):
    a = a / T
    return np.exp(a) / np.sum(np.exp(a))


a = np.array([3, 0, 1])
T_ = [100, 2, 1]

for (i, T) in enumerate(T_):
    plt.subplot(1, 3, i + 1)
    plt.bar([1, 2, 3], softmax(a, T))
    plt.axis([0, 4, 0, 1])
    plt.title('T = ' + str(T))
plt.show()
