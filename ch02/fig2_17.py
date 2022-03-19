import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma

a = [0.1, 0.1, 1, 2, 2]
b = [0.1, 1, 1, 2, 8]
x = np.linspace(0, 1, 100)
c = ['b', 'r', 'k', 'g', 'c']
for i in range(5):
    beta_pdf = beta.pdf(x, a[i], b[i])
    plt.plot(x, beta_pdf, c=c[i], label='a=' + str(a[i]) + ',b=' + str(b[i]))
plt.legend()
plt.show()

# plot gamma distribution
a = [1, 1.5, 2, 1, 1.5, 2]
b = [1, 1, 1, 2, 2, 2]
x = np.linspace(0, 7, 100)
c = ['b', 'r', 'k', 'b--', 'r--', 'k--']
for i in range(6):
    gamma_pdf = gamma.pdf(x, a[i], loc=0, scale=1 / b[i])
    plt.plot(x, gamma_pdf, c[i], label='a=' + str(a[i]) + ',b=' + str(b[i]))
plt.legend()
plt.show()
