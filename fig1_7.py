import matplotlib.pyplot as plt
import numpy as np


def make_1dregression_data(n=21):
    np.random.seed(0)
    xtrain = np.linspace(0.0, 20, n)
    xtest = np.arange(0.0, 20, 0.1)
    sigma2 = 4
    w = np.array([-1.5, 1 / 9.])
    fun = lambda x: w[0] * x + w[1] * np.square(x)
    ytrain = fun(xtrain) + np.random.normal(0, 1, xtrain.shape) * \
             np.sqrt(sigma2)
    ytest = fun(xtest) + np.random.normal(0, 1, xtest.shape) * \
            np.sqrt(sigma2)
    return xtrain, ytrain, xtest, ytest


xtrain, ytrain, xtest, ytest = make_1dregression_data(n=21)
# plt.scatter(xtrain, ytrain)
# plt.show()
# plt.scatter(xtest, ytest)
# plt.show()
xtrain = xtrain[:, np.newaxis] / 10 - 1
ytrain = ytrain[:, np.newaxis]

xtest = xtest[:, np.newaxis] / 10 - 1
ytest = ytest[:, np.newaxis]
print(xtrain.shape, ytrain.shape)


def poly_reg(X, y, X_test, degree=2):
    '''

    :param X: [N,p]
    :param y: [N,1]
    :param degree:
    :return:
    '''
    X_ = X ** 0
    X__ = X_test ** 2
    for i in range(1, degree + 1):
        X_ = np.concatenate([X_, X ** i], axis=1)
        X__ = np.concatenate([X__, X_test ** i], axis=1)
    # print(X_)
    # X_=X_[:,::-1]
    # X__ = X__[:, ::-1]
    w_est = np.linalg.pinv(X_.T @ X_) @ X_.T @ y
    # xxx = X_.T @ X_
    # print(np.max(xxx), np.min(xxx))
    # r = np.linalg.matrix_rank(xxx)
    # print(r)
    y_est = X_ @ w_est
    # print(X__.shape)
    y_test_est = X__ @ w_est
    return y_est, w_est, y_test_est


y_est2, w, _ = poly_reg(xtrain, ytrain, xtest, degree=2)
y_est14, w, _ = poly_reg(xtrain, ytrain, xtest, degree=14)
y_est20, w, _ = poly_reg(xtrain, ytrain, xtest, degree=20)
print(y_est2.shape)
# plt.subplot()
plt.scatter(xtrain, ytrain)
plt.plot(xtrain, y_est2, c='r')
plt.show()

plt.scatter(xtrain, ytrain)
plt.plot(xtrain, y_est14, c='r')
plt.show()

plt.scatter(xtrain, ytrain)
plt.plot(xtrain, y_est20, c='r')
plt.show()

err = []
err_test = []
for degree in range(1, 21):
    y_est, w, y_test_est = poly_reg(xtrain, ytrain, xtest, degree=degree)
    err.append(np.mean((y_est - ytrain) ** 2))
    err_test.append(np.mean((y_test_est - ytest) ** 2))
plt.plot(np.arange(20) + 1, err, '*-')
plt.plot(np.arange(20) + 1, err_test, '*-')
plt.axis([0, 20, 0, 100])
plt.show()
print(err_test, np.min(err_test))
