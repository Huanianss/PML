import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

np.random.seed(2)
mu = [0.5, 0.5]
Sigma = [[0.2, 0.1], [0.1, 0.1]]
N = 10
data = np.random.multivariate_normal(mu, Sigma, N)
print(data.shape)

plt.scatter(data[:, 0], data[:, 1])
plt.scatter(mu[0], mu[1], s=500, c='k', marker='x')
plt.axis([-1, 1, -1, 1])
plt.show()

# prior
Sigma_z = np.array([[0.1, 0], [0, 0.1]])
prior = multivariate_normal([0, 0], Sigma_z)
x = np.linspace(-1, 1, 100)
xx, yy = np.meshgrid(x, x)
x_ = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)

y_ = prior.pdf(x_)
y_ = y_.reshape(xx.shape)

plt.contour(xx, yy, y_)
plt.title('prior')
plt.show()

# posterior
Sigma_y = np.cov(data.T)

Sigma_inv = np.linalg.inv(Sigma_z) + N * np.linalg.inv(Sigma_y)
Sigma_post = np.linalg.inv(Sigma_inv)

mu_post = Sigma_post @ (np.linalg.inv(Sigma_y) @ np.mean(data.T, 1) * N)
print(mu_post)

posterior = multivariate_normal(mu_post, Sigma_post)
x = np.linspace(-1, 1, 100)
xx, yy = np.meshgrid(x, x)
x_ = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)

y_ = posterior.pdf(x_)
y_ = y_.reshape(xx.shape)

plt.contour(xx, yy, y_)
plt.title('posterior')
plt.show()
