import matplotlib.pyplot as plt
from scipy.stats import dirichlet

alpha = [0.1, 0.1, 0.1, 0.1, 0.1]
d1 = dirichlet(alpha)
data = d1.rvs(5)
for i in range(5):
    plt.subplot(5, 1, i + 1)
    plt.bar(range(5), data[i])
plt.show()

alpha = [1, 1, 1, 1, 1]
d1 = dirichlet(alpha)
data = d1.rvs(5)
for i in range(5):
    plt.subplot(5, 1, i + 1)
    plt.bar(range(5), data[i])
plt.show()
