import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# @title Synthesize dataset.
w0 = 0.125
b0 = 5.
x_range = [-20, 60]


def load_dataset(n=150, n_tst=150):
    np.random.seed(43)

    def s(x):
        g = (x - x_range[0]) / (x_range[1] - x_range[0])
        return 3 * (0.25 + g ** 2.)

    x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
    eps = np.random.randn(n) * s(x)
    y = (w0 * x * (1. + np.sin(x)) + b0) + eps
    x = x[..., np.newaxis]
    x_tst = np.linspace(*x_range, num=n_tst).astype(np.float32)
    x_tst = x_tst[..., np.newaxis]
    return y, x, x_tst


y, x, x_tst = load_dataset()
x = x / 100
y = y / 100
x_tst = x_tst / 100
y = y.reshape(150, 1)
print(y.shape, x.shape, x_tst.shape)

lr = LinearRegression()
lr.fit(x, y)
y_est = lr.predict(x_tst)

print(
    lr.coef_,
    lr.intercept_)

w = lr.coef_[0]
b = lr.intercept_

err = (w * x + b - y) ** 2
err = np.sum(err)
std = np.sqrt(err / 150)
print(std)

plt.scatter(x, y)
plt.plot(x_tst, y_est, 'r')
plt.plot(x_tst, y_est + std, 'g')
plt.plot(x_tst, y_est - std, 'g')
plt.show()
