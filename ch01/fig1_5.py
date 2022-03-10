import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
N = 21
x = np.linspace(0.0, 20, N)
X0 = x.reshape(N, 1)
X = np.c_[np.ones((N, 1)), X0]
w = np.array([-1.5, 1 / 9.])
y = w[0] * x + w[1] * np.square(x)
y = y + np.random.normal(0, 1, N) * 2
y = y[:, np.newaxis]

plt.scatter(x, y)
plt.show()

print(X.shape, y.shape)
print(X)

# normal equation
# Xw=y --> X'Xw=X'y --> w=inv(X'X)X'y

w_est = np.linalg.inv(X.T @ X) @ X.T @ y
print(w_est)

y_est = X @ w_est
plt.scatter(x, y)
plt.plot(x, y_est,'r')
plt.scatter(x, y_est, marker='x', s=50)
for i in range(21):
    plt.plot([x[i], x[i]], [y_est[i], y[i]], c='k')
plt.show()
