# from fig4_7
import matplotlib.pyplot as plt
import numpy as np


def poly_reg(X, y, X_test, lam=0, degree=2):
    '''

    :param X: [N,p]
    :param y: [N,1]
    :param degree:
    :return:
    '''
    X_ = X ** 0
    X__ = X_test ** 0
    for i in range(1, degree + 1):
        X_ = np.concatenate([X_, X ** i], axis=1)
        X__ = np.concatenate([X__, X_test ** i], axis=1)

    w_est = np.linalg.pinv(X_.T @ X_ + lam * np.eye(X_.shape[1])) @ X_.T @ y
    y_est = X_ @ w_est
    y_test_est = X__ @ w_est
    return y_est, w_est, y_test_est


N = 20
X = np.linspace(0, 1, 20).reshape([-1, 1])
y_true = np.cos(2 * np.pi * X)

Y_est = []
Y_est1 = []
lam1 = 1
lam2 = 0.0001
for i in range(N):
    y = y_true + np.random.randn(*X.shape) * 0.1
    y_est, _, _ = poly_reg(X, y, X, lam=lam1, degree=10)
    y_est1, _, _ = poly_reg(X, y, X, lam=lam2, degree=10)
    Y_est.append(y_est)
    Y_est1.append(y_est1)
Y_est = np.concatenate(Y_est, 1)
Y_est1 = np.concatenate(Y_est1, 1)
print(Y_est.shape)
plt.plot(X, Y_est, 'r')
plt.title('$\lambda$=' + str(lam1))
plt.show()
plt.plot(X, Y_est1, 'r')
plt.title('$\lambda$=' + str(lam2))
plt.show()

plt.plot(X, y_true, 'g', lw=2)
plt.plot(X, np.mean(Y_est, 1), 'r:', lw=2)
plt.title('$\lambda$=' + str(lam1))
plt.show()

plt.plot(X, y_true, 'g', lw=2)
plt.plot(X, np.mean(Y_est1, 1), 'r:', lw=2)
plt.title('$\lambda$=' + str(lam2))
plt.show()
