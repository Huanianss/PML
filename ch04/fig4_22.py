import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, norm
from scipy.special import comb, gamma

a, b = 2, 2
prior = beta(a, b)
N1, N0 = 1, 10
likelihood = beta(N1 + 1, N0 + 1)
posterior = beta(a + N1, b + N0)
x = np.linspace(0, 1, 100)

theta = np.linspace(0, 1, 21)
gridApproxi = theta ** N1 * (1 - theta) ** N0 * prior.pdf(theta) / np.sum(
    (theta ** N1 * (1 - theta) ** N0) * prior.pdf(theta))

plt.plot(x, prior.pdf(x), label='Prior')
plt.plot(x, likelihood.pdf(x), label='Likelihood')
plt.plot(x, posterior.pdf(x), label='Posterior')
plt.bar(theta, gridApproxi / 0.05, width=0.03, alpha=0.75, label='Grid Approxi')
plt.legend()
plt.show()

mu = (N1 - 1 + a) / (N0 + N1 + a + b - 2)
gradgrad = +(N1 + a - 1) / mu ** 2 + (b - 1 + N0) / (1 - mu) ** 2
std = +1 / np.sqrt(gradgrad)
print('Laplace Approximation ')
print("mu = {:.3f},std = {:.3f}".format(mu, std))


norm1 = norm(mu, std)

plt.plot(x, norm1.pdf(x), label='Laplace')
plt.plot(x, posterior.pdf(x), label='Posterior')
plt.legend()
plt.show()

print('*'*70)
print('max value of PDF (Numerical)')
print('Posterior = {:.3f},  Laplace Approximation = {:.3f}'.format(np.max(posterior.pdf(x)),np.max(norm1.pdf(x))) )


a = a + N1
b = b + N0

C = gamma(a) * gamma(b) / gamma(a + b)

pdf_beta_mu = mu ** (a - 1) * (1 - mu) ** (b - 1) / C

pdf_gaussian_mu = 1 / np.sqrt(2 * np.pi * std ** 2)

print('*'*70)
print('max value of PDF (Analytical)')
print('Posterior = {:.3f},  Laplace Approximation = {:.3f}'.format(pdf_beta_mu, pdf_gaussian_mu))
