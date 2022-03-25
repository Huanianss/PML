# https://zhuanlan.zhihu.com/p/93513123
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

mnist = torchvision.datasets.MNIST('../data', train=False, transform=torchvision.transforms.ToTensor())

N = 500
k = 20
img = mnist.data[0:N]

img = img.float() / 255
img = img.round()
img = img.reshape(N, 28 * 28).numpy()
# print(img.shape)

pi_k = np.random.random([1, k])
pi_k = np.abs(pi_k)
pi_k = pi_k / np.sum(pi_k)

theta_k = np.ones([k, 28 * 28]) / k

for i in range(50):
    temp = [theta_k ** x * (1 - theta_k) ** (1 - x) for x in img]
    # print(temp[0].shape, len(temp))
    p_xn_thetak = [np.prod(x, 1) for x in temp]
    p_xn_thetak = np.array(p_xn_thetak)
    # print(p_xn_thetak.shape)
    flag = np.sum(p_xn_thetak, axis=1)
    flag = np.sum(np.log(flag))
    print(i, flag)
    gamma_znk = pi_k * p_xn_thetak / np.sum(pi_k * p_xn_thetak, 1, keepdims=True)

    # print(gamma_znk.shape)
    theta = img.T @ gamma_znk / np.sum(gamma_znk, 0, keepdims=True)
    theta_k = theta.T

    pi_k = np.sum(gamma_znk, 0, keepdims=True) / N

print(theta_k.shape)
print(pi_k.shape)
print(img.shape)
means = np.reshape(theta_k, [k, 28, 28])
img = np.reshape(img, [N, 28, 28])
for i in range(4):
    for j in range(5):
        plt.subplot(4, 5, i * 5 + j + 1)
        plt.imshow(means[i * 5 + j])
        # plt.imshow(img[i * 5 + j])
        plt.title(np.round(pi_k[0, i * 5 + j], 3))
        plt.clim(0, 1)
        plt.axis('off')
plt.show()
