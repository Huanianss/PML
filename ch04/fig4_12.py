import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from scipy.special import comb, gamma

M = 10
prior_para = [1, 1]
N1 = 4
N0 = 1
alpha = prior_para[0] + N1
beta = prior_para[1] + N0
r'''
f(x, a, b) = \frac{\Gamma(a+b) x^{a-1} (1-x)^{b-1}}
                          {\Gamma(a) \Gamma(b)}
                          '''

# print(gamma(alpha), gamma(beta), gamma(alpha + beta))
Bb = []
for x in range(M + 1):
    numerator = gamma(alpha + x) * gamma(beta + M - x) / gamma(alpha + beta + M)
    denominator = gamma(alpha) * gamma(beta) / gamma(alpha + beta)

    Bb.append(comb(M, x) * numerator / denominator)

x = np.arange(0, M + 1)
plt.bar(x, Bb)
plt.title('posterior predictive')
plt.show()

bi = binom(M, N1 / (N1 + N0))
bi_pdf = bi.pmf(x)
plt.bar(x, bi_pdf)
plt.title('MAP predictive')
plt.show()
