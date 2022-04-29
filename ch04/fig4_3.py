import matplotlib.pyplot as plt
import numpy as np

N = 40
x = np.linspace(0, 160, N // 2)
y1 = 1 / 16 * x - 9
y2 = -0.0002 * (x - 100) ** 2 - 9


y = np.stack([y1, y2], axis=1)
y = y.reshape([N, 1])

beta = 0.99
mu = np.zeros([N])
mu_corr = np.zeros([N])
for t in range(N):
    if t == 0:
        mu[t] = 0
        mu_corr[t] = y[0]
    else:
        mu[t] = beta * mu[t - 1] + (1 - beta) * y[t]
        mu_corr[t] = beta * mu_corr[t - 1] + (1 - beta) * y[t]


plt.plot(y, markersize=8, marker='.')
plt.plot(mu, )
plt.plot(mu_corr, )
plt.title('beta = '+ str(beta))
plt.show()
