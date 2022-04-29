import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

x = np.linspace(0, 1, 200)
beta1 = beta(3, 9)
pdf = beta1.pdf(x)
cdf = beta1.cdf(x)
alpha = 0.05

index1 = np.argmin(np.abs(cdf - alpha / 2))
index2 = np.argmin(np.abs(cdf - (1 - alpha / 2)))
a = x[index1]
b = x[index2]
print(a, b)

#  Central interval
plt.plot(x, pdf, 'k')
plt.plot([a, a], [0, beta1.pdf(a)], 'b')
plt.plot([b, b], [0, beta1.pdf(b)], 'b')
plt.plot([a, b], [beta1.pdf(a), beta1.pdf(b)], 'b')
plt.axis([0, 1, 0, 3.5])
plt.title('CI')
plt.show()

# highest posterior density

a = 0.04
b = 0.48
hpd = beta1.cdf(a) + 1 - beta1.cdf(b)
print(hpd)

plt.plot(x, pdf, 'k')
plt.plot([a, a], [0, beta1.pdf(a)], 'b')
plt.plot([b, b], [0, beta1.pdf(b)], 'b')
plt.plot([a, b], [beta1.pdf(a), beta1.pdf(b)], 'b')
plt.axis([0, 1, 0, 3.5])
plt.title('HPD')
plt.show()
