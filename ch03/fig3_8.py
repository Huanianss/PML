import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 50)
var_pr = 5
prior = norm(0, np.sqrt(var_pr))
epsilon = 1
likelihood = norm(3, epsilon)

mu = 3 / epsilon / (1 / var_pr + 1 / epsilon)
sigma = 1 / (1 / var_pr + 1 / epsilon)
posterior = norm(mu, np.sqrt(sigma))

plt.plot(prior.pdf(x), 'b')
plt.plot(likelihood.pdf(x), 'r--')
plt.plot(posterior.pdf(x), 'k--')
plt.legend(['prior', 'likelihood', 'posterior'])
plt.show()
