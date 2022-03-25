import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def plotEllipse(mu, sigma):
    lam1, u = np.linalg.eig(sigma)
    lam1 = np.sqrt(lam1)

    theta0 = np.arctan(u[0, 1] / u[0, 0])
    theta = np.arange(0, 2 * np.pi, 0.01)
    x = lam1[0] * np.cos(theta)
    y = lam1[1] * np.sin(theta)

    data = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)

    Transform = np.array([[np.cos(theta0), -np.sin(theta0)],
                          [np.sin(theta0), np.cos(theta0)]])
    data = data @ Transform
    plt.plot(mu[0] + data[:, 0], mu[1] + data[:, 1])

if __name__ == '__main__':
    y1 = np.array([0, -1])
    y2 = np.array([1, 0])
    sigma1 = 0.05 * np.eye(2)
    sigma2 = 0.01 * np.eye(2)

    # sigma1 = np.array([[0.1, 0.01],
    #                    [0.01, 0.01]])
    # sigma2 = np.array([[0.01, 0.01],
    #                    [0.01, 0.1]])

    Lambda = np.linalg.pinv(sigma1) + np.linalg.pinv(sigma2)
    Sigma_post = np.linalg.pinv(Lambda)

    Mu_post = y1.T @ np.linalg.pinv(sigma1) + y2.T @ np.linalg.pinv(sigma2)
    Mu_post = Mu_post @ Sigma_post

    plotEllipse(y1, sigma1)
    plotEllipse(y2, sigma2)
    plotEllipse(Mu_post, Sigma_post)
    plt.show()
