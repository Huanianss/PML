import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0)
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
print(y.shape, x.shape, x_tst.shape)

print(y.dtype, x.dtype, x_tst.dtype)

x_ = torch.tensor(x / 100, dtype=torch.float32)
y_ = torch.tensor(y / 100, dtype=torch.float32).reshape(150, 1)
x_tst_ = torch.tensor(x_tst / 100, dtype=torch.float32)


class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.w = nn.Parameter(torch.Tensor(1, 1))
        self.b = nn.Parameter(torch.Tensor(1, 1))
        self.w1 = nn.Parameter(torch.Tensor(1, 1))
        self.b1 = nn.Parameter(torch.Tensor(1, 1))

        # self.log_var = nn.Parameter(torch.Tensor(1, 1))
        # nn.init.normal_(self.log_var)
        nn.init.normal_(self.w)
        nn.init.normal_(self.w1)
        nn.init.normal_(self.b)
        nn.init.normal_(self.b1)

    def forward(self, x, y):
        sigma_2 = torch.log(1 + torch.exp(self.w1 * x+self.b1))
        # sigma_2 = torch.exp(self.log_var)
        nll = torch.log(2 * np.pi * sigma_2) \
              + (y - self.w * x - self.b) ** 2 / sigma_2
        nll = torch.mean(nll)
        return nll


model = LR()
optim = torch.optim.Adam(model.parameters(), lr=0.01)

for i in range(2000):
    loss = model(x_, y_)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % 100 == 0:
        print(i, loss)

with torch.no_grad():
    y_pred = x_tst_ * model.w + model.b

    sigma_2 = torch.log(1 + torch.exp(model.w1 * x_tst_+model.b1))
    # sigma_2 = torch.exp(model.log_var)
    std = sigma_2.sqrt()




plt.scatter(x_, y_)
plt.plot(x_tst_, y_pred, 'r',lw=4)
plt.plot(x_tst_, y_pred + std, 'g',lw=2)
plt.plot(x_tst_, y_pred - std, 'g',lw=2)
plt.show()

# print(std)
print(model.w, model.b)
print(model.w1, model.b)


# x=np.linspace(-30,30,100)
# y=np.log(1+np.exp(x))
# plt.plot(x,y)
# plt.show()
