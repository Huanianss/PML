import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

data = scio.loadmat('../data/anscombe.mat')
data = data['anscombe']
print(data.shape)


for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.scatter(data[0 + i * 11:11 + i * 11, 0], data[0 + i * 11:11 + i * 11, 1])
plt.show()

for i in range(4):
    x = data[0 + i * 11:11 + i * 11, 0]
    y = data[0 + i * 11:11 + i * 11, 1]

    my_rho = np.corrcoef(x, y)
    print('*' * 80)
    print('rho: ', my_rho)

    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    c = np.concatenate([x, y], axis=1)

    print('mean: ', np.mean(c, axis=0))
    print('cov: ', np.cov(c.T))