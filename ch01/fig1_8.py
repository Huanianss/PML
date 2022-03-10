from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target

plt.scatter(X[:, 2], X[:, 3], c='k')
plt.show()

from sklearn.mixture import GaussianMixture
gmm=GaussianMixture(3)
gmm.fit(X)
y_est=gmm.predict(X)
print(y_est)
plt.scatter(X[:, 2], X[:, 3], c=y_est)
plt.show()
