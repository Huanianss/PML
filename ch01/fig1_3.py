from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# load data
iris = load_iris()

X = iris.data
y = iris.target

print('X shape:', X.shape, 'y shape', y.shape)

plt.figure()
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, i * 4 + j + 1)
        if i is not j:
            plt.scatter(X[:, j], X[:, i], c=y, s=5)
        else:
            plt.hist(X[y == 0, j], 10, alpha=0.7)
            plt.hist(X[y == 1, j], 10, alpha=0.7)
            plt.hist(X[y == 2, j], 10, alpha=0.7)
plt.show()
