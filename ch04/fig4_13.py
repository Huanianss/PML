import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from scipy.special import comb, gamma


prior_para1 = [20, 20]
prior_para2 = [30, 10]
N1 = 80
N0 = 50

x = np.linspace(0, 1, 10000)
beta1 = beta(*prior_para1)
beta2 = beta(*prior_para2)
prior = 0.5 * beta1.pdf(x) + 0.5 * beta2.pdf(x)

# mixture component 1
a = prior_para1[0]
b = prior_para1[1]
denominator = gamma(a) * gamma(b) / gamma(a + b)

a = a + N1
b = b + N0
numerator = gamma(a) * gamma(b) / gamma(a + b)

beta1 = beta(a, b)
PD1 = comb(N1 + N0, N1) * numerator / denominator

# mixture component 2
a = prior_para2[0]
b = prior_para2[1]
denominator = gamma(a) * gamma(b) / gamma(a + b)

a = a + N1
b = b + N0
numerator = gamma(a) * gamma(b) / gamma(a + b)
PD2 = comb(N1 + N0, N1) * numerator / denominator
beta2 = beta(a, b)

ph1 = 0.5 * PD1 / (0.5 * PD1 + 0.5 * PD2)
ph2 = 0.5 * PD2 / (0.5 * PD1 + 0.5 * PD2)
print('mixture coefficient: ', ph1, ph2)
posterior = ph1 * beta1.pdf(x) + ph2 * beta2.pdf(x)


plt.plot(x, prior, 'r--', label='prior')
plt.plot(x, posterior, 'b', label='posterior')
plt.legend()
plt.show()

p_theta05 = sum(posterior[5000::]) * 0.0001
print('P(theta greater than 0.5)= ', p_theta05)
