# from fig1_7
import matplotlib.pyplot as plt
import numpy as np


def make_1dregression_data(n=21):
    np.random.seed(0)
    xtrain = np.linspace(0.0, 20, n)
    xtest = np.arange(0.0, 20, 1)
    sigma2 = 4
    w = np.array([-1.5, 1 / 9.])
    fun = lambda x: w[0] * x + w[1] * np.square(x)
    ytrain = fun(xtrain) + np.random.normal(0, 1, xtrain.shape) * \
             np.sqrt(sigma2)
    ytest = fun(xtest) + np.random.normal(0, 1, xtest.shape) * \
            np.sqrt(sigma2)
    return xtrain, ytrain, xtest, ytest


xtrain, ytrain, xtest, ytest = make_1dregression_data(n=200)

xtrain = xtrain[:, np.newaxis] / 10 - 1
ytrain = ytrain[:, np.newaxis]

xtest = xtest[:, np.newaxis] / 10 - 1
ytest = ytest[:, np.newaxis]

print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape)

train_data = np.concatenate([xtrain, ytrain], axis=1)
train_data = np.random.permutation(train_data)
xtrain = train_data[:, 0:1]
ytrain = train_data[:, 1:2]
# print(xtrain)
print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape)


# plt.scatter(xtrain,ytrain)
# plt.show()
# plt.scatter(xtest,ytest)
# plt.show()


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


degree_ = [1, 2, 10, 20]

for degree in degree_:

    err_train = []
    err_test = []
    for i in np.linspace(10, 200, 20):
        train_index = np.arange(0, i).astype(np.uint)
        X_train = xtrain[train_index]
        y_train = ytrain[train_index]

        y_est, w, y_test_est = poly_reg(X_train,
                                        y_train,
                                        xtest,
                                        lam=0,
                                        degree=degree)
        err_train.append(np.mean((y_est - y_train) ** 2))
        err_test.append(np.mean((y_test_est - ytest) ** 2))

    plt.plot(np.linspace(10, 200, 20),
             err_train,
             c='b',
             lw=3,
             marker='s',
             label='train',
             markersize=10,
             )
    plt.plot(np.linspace(10, 200, 20),
             err_test,
             c='r',
             lw=3,
             marker='x',
             label='test',
             markersize=10)
    plt.plot([0, 200], [4, 4], 'k', lw=3)
    plt.title('truth = degree 2, model = degree ' + str(degree))
    plt.legend(fontsize=20)
    plt.axis([0, 200, 0, 20])
    plt.show()
