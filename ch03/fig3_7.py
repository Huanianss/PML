import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import numpy as np
import torch

np.random.seed(0)
D = 10
N = 7
mu = np.zeros([D])
cov = np.random.randn(D, D)
cov = cov @ cov.T / 110
# cov=np.eye(D)
data = np.random.multivariate_normal(mu, cov, N)
data = data.T
print(data.shape)
print(np.min(data))


def plot_hinton_digram(data):
    for i in range(D):
        for j in range(N):
            s = data[i, j]
            if s >= 0:
                c = 'b'
            else:
                s = -s
                c = 'g'

            x1 = i - s / 2
            x2 = i + s / 2

            y1 = j - s / 2
            y2 = j + s / 2
            plt.fill_between([x1, x2], [y1, y1], [y2, y2], color=c)
    plt.grid('on')
    plt.show()


plot_hinton_digram(data)

mask = np.concatenate([np.ones([D // 2, N]), np.zeros([D // 2, N])], axis=0)
mask = [np.random.permutation(m) for m in mask.T]
mask = np.concatenate(mask)
mask = mask.reshape([N, D]).T

data_o = data * mask
plot_hinton_digram(data_o)

mask = torch.tensor(mask)
for i in range(N):
    # print(data.shape, mask.shape)
    mask_permute, index = torch.sort(mask[:, i])
    cov_permute = cov[index.numpy(), :]
    cov_permute = cov_permute[:, index.numpy()]

    Lam = np.linalg.pinv(cov_permute)

    Lam_aa = Lam[0:D // 2, 0:D // 2]
    Lam_ab = Lam[0:D // 2, D // 2::]

    data_o_permute = data_o[index.numpy(), i]
    x_b = data_o_permute[D // 2::]

    # print(Lam_aa.shape, Lam_ab.shape, x_b.shape)
    mu_a_b = -np.linalg.inv(Lam_aa) @ Lam_ab @ x_b

    _, index = torch.sort(index)

    # mu_a_b x_b
    d = np.concatenate([mu_a_b, x_b])
    d = d[index]
    data_o[:, i] = d

plot_hinton_digram(data_o)
