import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


iris = load_iris()

X = iris.data
y = iris.target

class1 = X[y == 2][:, 3]
class2 = X[y != 2][:, 3]

y = y - 1
y = y > 0


lr = LogisticRegression()
lr.fit(X[:, 3:4], y)

xtest = np.linspace(0, 3, 50).reshape(-1, 1)
y_est = lr.predict_proba(xtest)

plt.scatter(class1, np.ones_like(class1), c='g', marker='^')
plt.scatter(class2, np.zeros_like(class2), c='b', marker='s')
plt.plot(xtest, y_est[:, 0], '--', c='b', label='Not Iris-Virginica')
plt.plot(xtest, y_est[:, 1], c='g', label='Iris-Virginica')
plt.plot([xtest[27], xtest[27]], [0, 1], 'k--')
plt.show()

print(xtest[27])