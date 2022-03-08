from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target

plt.scatter(X[:, 2], X[:, 3], c=y, s=10)
plt.plot([2.45, 2.45], [0, 2.5])
plt.plot([2.45, 7], [1.75, 1.75])
plt.fill_between([1, 2.45], [0, 0], [2.5, 2.5], alpha=0.3)
plt.fill_between([2.45, 7], [0, 0], [2.5, 2.5], alpha=0.3)
plt.fill_between([2.45, 7], [0, 0], [1.75, 1.75], alpha=0.3)
plt.axis([1, 7, 0, 2.5])
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()