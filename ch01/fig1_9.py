from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()

X = iris.data
y = iris.target

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
plt.show()

X_ =  X[:, 0:2]

y_ = X[:, 2:3]
X_=np.concatenate([np.ones_like(y_),X_],axis=1)
print(X_.shape, y_.shape)

w_est = np.linalg.inv(X_.T @ X_) @ X_.T @ y_
print(w_est.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='r')

x_=np.linspace(4,8,10)
y_=np.linspace(2,5,10)
xx, yy = np.meshgrid(x_, y_)
x2_est=w_est[0, 0] +xx * w_est[1, 0] + yy * w_est[2, 0]
ax.scatter(xx, yy, x2_est, color='k')
plt.show()
