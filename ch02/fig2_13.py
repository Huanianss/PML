import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()

X = iris.data
y = iris.target

lr = LogisticRegression()
lr.fit(X[:, 2:4], y)
x_ = np.linspace(1, 7, 100)
y_ = np.linspace(-1, 4, 100)
xx, yy = np.meshgrid(x_, y_)

xtest = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
y_est = lr.predict_proba(xtest)
y_est1 = y_est[:, 1].reshape(100, 100)

y_pred = np.argmax(y_est, 1)
y_pred = y_pred.reshape(100, 100)

plt.figure()
plt.contourf(xx, yy, y_pred, alpha=0.3)
plt.scatter(X[:, 2], X[:, 3], c=y)
c = plt.contour(xx, yy, y_est1)
plt.clabel(c)
plt.show()
