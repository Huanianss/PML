import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
import seaborn as sns

# load data
iris = load_iris()

X = iris.data
y = iris.target

Cov = np.cov(X.T)
Corr = np.corrcoef(X.T)

sns.heatmap(Cov,
            annot=True,
            cmap='viridis',
            yticklabels=iris.feature_names)

plt.show()
mask = 1 - np.tri(4) + np.eye(4)
# print(mask)
sns.heatmap(Corr,
            annot=True,
            cmap='viridis',
            mask=mask,
            yticklabels=iris.feature_names)
plt.show()

print('Cov:', Cov)
print('Corr: ', Corr)

N = X.shape[0]
C_N = np.eye(N) - 1 / N * np.ones([N, 1]) @ np.ones([1, N])

Cov1 = 1 / N * X.T @ C_N @ X
Corr1 = np.diag(np.diag(Cov1) ** -0.5) @ Cov1 @ np.diag(np.diag(Cov1) ** -0.5)
print('Cov1 :', Cov1)
print('Corr :', Corr1)



