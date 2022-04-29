import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, binom

prior_para_list = [(2, 2),
                   (1, 1)]
N1 = 4
N0 = 1

x = np.linspace(0, 1, 50)
posterior_para = [[], []]
for prior_para in prior_para_list:
    prior = beta(prior_para[0], prior_para[1])
    likelihood = beta(N1 + 1, N0 + 1)

    posterior_para[0] = prior_para[0] + N1
    posterior_para[1] = prior_para[1] + N0
    posterior = beta(posterior_para[0], posterior_para[1])

    plt.plot(x, prior.pdf(x), 'r', label='prior Be({},{})'.format(*prior_para))
    plt.plot(x, likelihood.pdf(x), 'k--', lw=4, label='lik Be({},{})'.format(N1 + 1, N0 + 1))
    plt.plot(x, posterior.pdf(x), 'b-.', label='post Be({},{})'.format(*posterior_para))
    plt.legend()
    plt.show()
