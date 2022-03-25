import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

mu1 = np.array([0.8, 0.6])
mu2 = np.array([0.5, 0.5])
mu3 = np.array([0.2, 0.4])

sigma1 = 0.01 * np.array([[1, -0.6],
                          [-0.6, 1]])

sigma2 = 0.01 * np.array([[1, 0.6],
                          [0.6, 1]])

sigma3 = sigma1
# plotEllipse(mu1,sigma1)
# plotEllipse(mu2,sigma2)
# plotEllipse(mu3,sigma3)
# plt.show()

rv1 = multivariate_normal(mu1, sigma1)
rv2 = multivariate_normal(mu2, sigma2)
rv3 = multivariate_normal(mu3, sigma3)

x = np.linspace(0, 1, 100)
xx, yy = np.meshgrid(x, x)
x_ = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
y_ = rv1.pdf(x_)
y1 = y_.reshape(xx.shape)
plt.contour(xx, yy, y1)

y_ = rv2.pdf(x_)
y2 = y_.reshape(xx.shape)
plt.contour(xx, yy, y2)

y_ = rv3.pdf(x_)
y3 = y_.reshape(xx.shape)
plt.contour(xx, yy, y3)

plt.show()


pdf=0.2*y1+0.2*y2+0.6*y3
ax = plt.subplot(111, projection='3d')
ax.plot_surface(xx, yy, pdf, cmap='jet')
plt.show()
