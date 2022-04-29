# from fig1_7
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold


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


xtrain, ytrain, xtest, ytest = make_1dregression_data(n=80)

xtrain = xtrain[:, np.newaxis] / 10 - 1
ytrain = ytrain[:, np.newaxis]

xtest = xtest[:, np.newaxis] / 10 - 1
ytest = ytest[:, np.newaxis]

print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape)




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


kf = KFold(n_splits=5, shuffle=True)
err = []
for train_index, test_index in kf.split(xtrain):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = xtrain[train_index], xtrain[test_index]
    y_train, y_test = ytrain[train_index], ytrain[test_index]

    lam_ = np.logspace(-10, 2, 13)
    for i in range(13):
        lam = lam_[i]
        y_est, w, y_test_est = poly_reg(X_train,
                                        y_train,
                                        X_test,
                                        lam=lam,
                                        degree=14)
        err.append(np.mean((y_test_est - y_test) ** 2))

err = np.array(err)

err = err.reshape(5, 13)

err_mean = np.mean(err, 0)
err_std = np.std(err, 0)

print(err_mean)

plt.errorbar(np.linspace(-10, 2, 13), err_mean, err_std)
plt.show()
