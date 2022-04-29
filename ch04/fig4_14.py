import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import dirichlet

N = 50
x = np.linspace(0.01, 1, N)
y = x
xx, yy = np.meshgrid(x, y)
zz = 1 - xx - yy

zz[zz < 0] = None

plt.figure()
ax = plt.subplot(111, projection='3d')

mask = np.tri(N)  # + np.eye(N)
mask = mask[::-1]
# print(mask)
ax.plot_surface(xx * mask, yy * mask, zz)
plt.axis([0, 1, 0, 1])
plt.xlabel('x')
plt.ylabel('y')
ax.view_init(30, 20)
plt.show()

alpha_ = np.array([[20, 20, 20],
                   [3, 3, 20],
                   [0.1, 0.1, 0.1]])
for alpha in alpha_:
    d1 = dirichlet(alpha)

    x_ = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], axis=1)
    # print(x_.shape)
    y_ = d1.pdf(x_.T)
    y_ = y_.T.reshape(xx.shape)

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(xx, yy, y_, vmin=0, vmax=3, cmap='jet')
    ax.view_init(30, 20)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()
