import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, bernoulli

theta = 0.7
N = 10
bins = int(1/9*(N-10)+10)


posterior_nums = 10000
B = 10000

data = bernoulli(theta).rvs(N)
print(data)
N1 = np.sum(data)
N0 = N - N1

# Bootstrap approximation of the sampling distribution
theta_hat = N1 / N
theta_S = []
for i in range(B):
    data_s = bernoulli(theta).rvs(N)
    theta_S.append(np.sum(data_s) / N)

# a[i]=np.sum(bernoulli(theta).rvs(N))/N

plt.hist(theta_S, bins=bins)
plt.title('Boot:true = ' + str(theta) + ', n = ' + str(N) + ', mle = ' + str(theta_hat))

plt.show()

# Bayes
a = 1
b = 1
posterior_a = N1 + a
posterior_b = N0 + b
posterior_samples = beta(posterior_a, posterior_b).rvs(posterior_nums)
posterior_mean = np.round(np.mean(posterior_samples), 2)
plt.hist(posterior_samples, bins=bins)
plt.title('Bayes:true = ' + str(theta) + ' n = ' + str(N) + ', post mean = ' + str(posterior_mean))
plt.show()
