from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import rv_discrete

# load data
iris = load_iris()

X = iris.data
y = iris.target



X = X[y < 2, 0:1]
y = y[y < 2]

x1 = X[y == 0, 0]
y1 = np.zeros_like(x1)
y1 = y1 + np.random.randn(*y1.shape) * 0.01
x2 = X[y == 1, 0]
y2 = np.ones_like(x2)
y2 = y2 + np.random.randn(*y2.shape) * 0.01

lr = LogisticRegression()
lr.fit(X, y)
x_pred = np.linspace(4, 7, 100).reshape(-1, 1)
y_pred = lr.predict_proba(x_pred)

index = np.argmin(np.abs(y_pred[:, 0] - 0.5))
print(index)
print('decision boundary ', x_pred[index])
print('model weight',lr.coef_, lr.intercept_)

plt.figure()
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.plot(x_pred, y_pred[:, 1])
plt.plot([x_pred[index], x_pred[index]], [0, 1], 'k')
plt.xlabel(iris.feature_names[0])
plt.ylabel('p(y)=1')
plt.show()


N = 441
w = np.linspace(-40, 40, N)
b = np.linspace(-40, 40, N)
w, b = np.meshgrid(w, b)
theta = np.concatenate([w.reshape([1, -1]), b.reshape([1, -1])])
X = np.concatenate([X, np.ones_like(X)], axis=1)
sigma = 1 / (1 + np.exp(-X @ theta))
y = y.reshape([-1, 1])
ll = sigma ** y * (1 - sigma) ** (1 - y)
ll = np.prod(ll, 0)
posterior = ll / np.sum(ll, 0)
#
# print('theta', theta.shape)

# print('pos', posterior.shape)
# print(posterior)
# print(np.sum(posterior))
# a=np.arange(0,N)
# print('a',a.shape)
theta_S = np.random.choice(np.arange(0, N * N), N, p=posterior)
theta_S = theta[:, theta_S]
# print(theta_S.shape)

yy = 1 / (1 + np.exp(-theta_S[1, :] - x_pred * theta_S[0, :]))
# print(x_pred.shape, yy.shape)
y_pred_mean = np.mean(yy, 1)
y_pred_std = np.std(yy, 1)

x_star = -theta_S[1, :] / theta_S[0, :]
x_star_mean = np.mean(x_star)
x_star_std = np.std(x_star)

posteriorPlot = posterior.reshape([N, N])
plt.figure()
ax = plt.subplot(111, projection='3d')
ax.plot_surface(w, b, posteriorPlot, cmap='jet')
ax.view_init(-90, 0)
plt.xlabel('w')
plt.ylabel('b')
plt.show()


print(x_pred.shape)
print(y_pred_mean.shape)
print(y_pred_std.shape)
plt.figure()
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.plot(x_pred, y_pred_mean,'g')
plt.fill_between(x_pred[:,0],
                 y_pred_mean - y_pred_std,
                 y_pred_mean + y_pred_std,
                 color='g',
                 alpha=0.3)

plt.plot([x_star_mean, x_star_mean], [0, 1], 'k')
plt.fill_between([x_star_mean - x_star_std, x_star_mean + x_star_std],
                 [0, 0],
                 [1, 1],
                 color='k',
                 alpha=0.3)
plt.xlabel(iris.feature_names[0])
plt.ylabel('p(y)=1')
plt.show()
