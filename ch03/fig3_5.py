import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

mu = [0, 0]
Covs = {'Full': [[2, 1.8], [1.8, 2]],
        'Diag': [[1, 0], [0, 3]],
        'Spherical': [[1, 0], [0, 1]]}


def plot_norm_pdf_2d(sigma):
    norm1 = multivariate_normal(mu, sigma)
    x = np.linspace(-5, 5, 1000)
    xx, yy = np.meshgrid(x, x)

    x_ = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
    y_ = norm1.pdf(x_)
    y_ = y_.reshape(xx.shape)
    print(y_.shape)
    plt.figure()
    ax = plt.subplot(121, projection='3d')
    ax.plot_surface(xx, yy, y_, cmap='jet')

    ax = plt.subplot(122, projection='3d')
    plt.contour(xx, yy, y_)
    ax.view_init(90, 0)




for key in Covs:
    sigma = Covs[key]
    plot_norm_pdf_2d(sigma)
plt.show()
