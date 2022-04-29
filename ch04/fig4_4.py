import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
N = 100
lam = 0.9
d = 50
mu = np.zeros([d])
Sigma = np.random.randn(d, d)
Sigma = Sigma @ Sigma.T + 100 * np.eye(d)


c_true = np.linalg.cond(Sigma, 2)

x = np.random.multivariate_normal(mu, Sigma, N)

Cov = np.cov(x.T)
c_cov = np.linalg.cond(Cov, 2)

Cov_map = (1 - lam) * Cov + np.eye(d) * Cov * lam
c_cov_map = np.linalg.cond(Cov_map, 2)

_, lam_true, _ = np.linalg.svd(Sigma)
_, lam_cov, _ = np.linalg.svd(Cov)
_, lam_cov_map, _ = np.linalg.svd(Cov_map)

plt.plot(np.arange(d), lam_true, 'k', label='true, k=' + str(round(c_true)))
plt.plot(np.arange(d), lam_cov, 'b.', label='mle, k=' + str(round(c_cov)))
plt.plot(np.arange(d), lam_cov_map, 'r.-', label='map, k=' + str(round(c_cov_map)))
plt.legend()
plt.title('N = ' + str(N))
plt.show()
