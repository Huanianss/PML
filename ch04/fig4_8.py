# from fig1_7
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

xtrain = xtrain[:, np.newaxis] / 10 - 1
ytrain = ytrain[:, np.newaxis]

xtest = xtest[:, np.newaxis] / 10 - 1
ytest = ytest[:, np.newaxis]

print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape)


def poly_reg_grad(X, y, X_test, y_test, degree=2):
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

    w = np.random.randn(degree + 1, 1)
    err_train = []
    err_test = []
    for i in range(20):
        w_grad = X_.T @ X_ @ w - X_.T @ y
        w = w - 0.004 * w_grad
        y_est = X_ @ w
        y_test_est = X__ @ w

        err_train.append(np.mean(y_est - y) ** 2)
        err_test.append(np.mean(y_test_est - y_test) ** 2)
    return err_train, err_test


err_train, err_test = poly_reg_grad(xtest,
                                    ytest,
                                    xtrain,
                                    ytrain,
                                    degree=8)

plt.plot(err_train, label='train MSE')
plt.plot(err_test, label='test MSE')
plt.legend()
plt.xlabel('iter')
plt.ylabel('MSE')
plt.show()
