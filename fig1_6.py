from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

data = loadmat('./data/moteData.mat')
X = data['X']
y = data['y']



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='r')
plt.show()
print(X.shape, y.shape)

X_ = np.concatenate([np.ones_like(y), X], axis=1)
print(X_.shape)
w_est = np.linalg.inv(X_.T @ X_) @ X_.T @ y
print(w_est)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='r')
xx, yy = np.meshgrid(X[:, 0], X[:, 1])
ax.plot_surface(xx, yy, w_est[0, 0] + xx * w_est[1, 0] + yy * w_est[2, 0], cmap='jet')
plt.show()

X_ = np.concatenate([np.ones_like(y), X, X ** 2], axis=1)
print(X_.shape)
w_est = np.linalg.inv(X_.T @ X_) @ X_.T @ y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='r')
xx, yy = np.meshgrid(X[:, 0], X[:, 1])
ax.plot_surface(xx, yy,
                w_est[0, 0] + xx * w_est[1, 0] + yy * w_est[2, 0] + xx ** 2 * w_est[3, 0] + yy ** 2 * w_est[4, 0],
                cmap='jet')
plt.show()
